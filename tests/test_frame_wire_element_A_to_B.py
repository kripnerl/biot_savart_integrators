from biot_savart_integrators.generic.line_elements import biot_savart_integral_A, biot_savart_integral_B

import numpy as np
import xarray as xr
from scipy.constants import pi, mu_0

import pytest


@pytest.fixture
def setup():
    """
    Setup frame coil in in x=0 plane.

    """
    I = 1.15766
    a = 1.0  # 1.23 # square side length
    N_coil = 1000

    # size of the cube N_cube * N_cube * M_cube (width * width * depth)
    N_cube = 32
    M_cube = 6
    width_cube = 0.3 # side length of the cube
    depth_cube = 0.1
    dist_cube = 0.3 # distance of the cube from the x=0 plane

    coil_edges = np.array([[0, a/2, -a/2],
                           [0, -a/2, -a/2],
                           [0, -a/2, a/2],
                           [0, a/2, a/2]])

    coil = list()

    for i in range(4):
        rs = np.linspace(coil_edges[i-1], coil_edges[i], N_coil, endpoint=False)
        coil.append(rs)

    coil = np.concatenate(coil)

    coil_da = xr.DataArray(coil,
                           dims=["idx", "dim"],
                           coords={"dim": list("xyz")})

    grid = xr.Dataset(coords={"x": dist_cube + np.linspace(0, depth_cube, M_cube),
                              "y": np.linspace(0, width_cube, N_cube),
                              "z": np.linspace(0, width_cube, N_cube)})
    grid = xr.concat(xr.broadcast(grid.x, grid.y, grid.z,),
                     dim="dim", coords="minimal").stack(index=["x", "y", "z"])
    grid.coords["dim"] = list("xyz")

    return coil_da, grid, I, "idx", "dim"


def A_to_B_cart(A, spacial_dim):
    Ax = A.sel(**{spacial_dim: "x"})
    Ay = A.sel(**{spacial_dim: "y"})
    Az = A.sel(**{spacial_dim: "z"})

    dAxdy = Ax.differentiate("y")
    dAxdz = Ax.differentiate("z")

    dAydx = Ay.differentiate("x")
    dAydz = Ay.differentiate("z")

    dAzdx = Az.differentiate("x")
    dAzdy = Az.differentiate("y")

    Bx = dAzdy - dAydz
    By = dAxdz - dAzdx
    Bz = dAydx - dAxdy

    return Bx, By, Bz


def test_rotA_toB(setup):

    coil, grid, I, int_dim, spatial_dim = setup

    A = biot_savart_integral_A(r=grid, r_c=coil, j=I, integration_dim=int_dim, spatial_dim=spatial_dim)
    A = A.unstack()
    B = biot_savart_integral_B(r=grid, r_c=coil, j=I, integration_dim=int_dim, spatial_dim=spatial_dim)
    B = B.unstack()

    Bx, _, _ = A_to_B_cart(A, spatial_dim)

    Bx2 = B.sel(**{spatial_dim: "x"})

    np.testing.assert_allclose(Bx.isel(x=2), Bx2.isel(x=2), atol=1e-4)
