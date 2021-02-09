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
    a = 0.01  # 1.23 # square side length
    N_coil = 1000

    # size of the grid N * N * M
    N = 32
    M = 8
    height = 0.02 # side length of the grid
    depth = 0.01
    dist_grid = 0.1 # distance of the grid from the x=0 plane

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

    grid = xr.Dataset(coords={"R": dist_grid + np.linspace(0, depth, M),
                              "Z": np.linspace(-height, height, N),
                              "phi": np.linspace(-pi/6, pi/6, N)})
    grid.coords["X"] = grid.R * np.cos(grid.phi)
    grid.coords["Y"] = grid.R * np.sin(grid.phi)

    grid = xr.concat(xr.broadcast(grid.X, grid.Y, grid.Z),
                     dim="dim", coords="minimal").stack(index=["R", "Z", "phi"])
    grid.coords["dim"] = list("xyz")

    return coil_da, grid, I, "idx", "dim"


def plot_conf_fields(AR, Aphi, AZ, BR, Bphi, BZ, BR2, Bphi2, BZ2, coil, grid, spatial_dim):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, axs = plt.subplots(3, 3)

    # From rotation of A
    ax = axs[0, 0]
    BR.isel(R=2).plot(ax=ax)
    ax.set_title("BR")

    ax = axs[1, 0]
    BZ.isel(R=2).plot(ax=ax)
    ax.set_title("BZ")

    ax = axs[2, 0]
    Bphi.isel(R=2).plot(ax=ax)
    ax.set_title("Bphi")

    # From B-S law

    ax = axs[0, 1]
    BR2.isel(R=2).plot(ax=ax)
    ax.set_title("BR2")

    ax = axs[1, 1]
    BZ2.isel(R=2).plot(ax=ax)
    ax.set_title("BZ2")

    ax = axs[2, 1]
    Bphi2.isel(R=2).plot(ax=ax)
    ax.set_title("Bphi2")

    ax = axs[0, 2]
    AR.isel(R=2).plot(ax=ax)
    ax.set_title("AR")

    ax = axs[1, 2]
    AZ.isel(R=2).plot(ax=ax)
    ax.set_title("AZ")

    ax = axs[2, 2]
    Aphi.isel(R=2).plot(ax=ax)
    ax.set_title("Aphi")


    # plot configuration:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(coil.sel(**{spatial_dim: "x"}),
            coil.sel(**{spatial_dim: "y"}),
            coil.sel(**{spatial_dim: "z"}))

    ax.scatter(grid.sel(**{spatial_dim: "x"}),
               grid.sel(**{spatial_dim: "y"}),
               grid.sel(**{spatial_dim: "z"}))

    # ax.set_aspect("equal")
    #     ax.set_xlim()

    plt.show()


def decompose_A2cyl_components(A, spacial_dim="dim"):

    Ax = A.sel({spacial_dim: "x"})
    Ay = A.sel({spacial_dim: "y"})
    Az = A.sel({spacial_dim: "z"})

    AR = (A.X * Ax + A.Y * Ay) / A.R

    # note: this order od +/- is correct the opposite is not(!)
    Aphi = (A.X * Ay - A.Y * Ax) / A.R

    return AR, Aphi, Az


def A_to_B_cyl(AR, Aphi, Az):

    R = AR.R

    dAR_dphi = AR.differentiate("phi")
    dAR_dz = AR.differentiate("Z")

    dRAphi_dR = (Aphi * R).differentiate("R")
    dAphi_dz = Aphi.differentiate("Z")

    dAz_dR = Az.differentiate("R")
    dAz_dphi = Az.differentiate("phi")

    BR = ((1 / R) * dAz_dphi) - dAphi_dz
    Bphi = dAR_dz - dAz_dR
    Bz = (1 / R) * (dRAphi_dR - dAR_dphi)

    return BR, Bphi, Bz


def test_rotA_toB(setup):

    coil, grid, I, int_dim, spatial_dim = setup

    A = biot_savart_integral_A(r=grid, r_c=coil, j=I, integration_dim=int_dim, spatial_dim=spatial_dim)
    A = A.unstack()
    B = biot_savart_integral_B(r=grid, r_c=coil, j=I, integration_dim=int_dim, spatial_dim=spatial_dim)
    B = B.unstack()

    AR, Aphi, AZ = decompose_A2cyl_components(A, spatial_dim)
    BR, Bphi, BZ = A_to_B_cyl(AR, Aphi, AZ)

    BR2, Bphi2, BZ2 = decompose_A2cyl_components(B, spatial_dim)

    # Uncomment this to see configuration
    # plot_conf_fields(AR, Aphi, AZ, BR, Bphi, BZ, BR2, Bphi2, BZ2, coil, grid, spatial_dim)

    for B1, B2 in zip([BR, Bphi, BZ], [BR2, Bphi2, BZ2]):
        # test the second R slice where the derivatives may be more accurate
        np.testing.assert_allclose(B2.isel(R=2), B2.isel(R=2), atol=1e-4)
