from biot_savart_integrators.generic import curve as generic_curve
from biot_savart_integrators.generic import line_elements

import numpy as np
import xarray as xr
from scipy.constants import pi, mu_0

import pytest


@pytest.fixture
def setup():
    I = 1.15766
    a = 1.0  # 1.23
    # high phi resolution to fulfill tolerance

    zs = xr.DataArray(np.linspace(-10000, 10000, int(1e6)), dims=['z'], name='z')
    # r0 = xr.DataArray([a, 0, 2.1], coords=[('s', list('xyz'))], name='r0')
    r0 = xr.DataArray([a, 0, 0], coords=[('s', list('xyz'))], name='r0')
    r_c = xr.concat([xr.zeros_like(zs), xr.zeros_like(zs), zs], r0.s)
    return I, a, r0, r_c


def test_inf_wire_line_el(setup):
    I, a, r0, r_c = setup
    dl = r_c.differentiate("z")  # vector
    j = dl * I  # vector with length element
    dl = (dl ** 2).sum("s") ** 0.5  # actual lengths
    j = j / dl  # normalization to line element length.

    Btheta_analytic = mu_0 * I / (2 * np.pi * a)

    B_line_el = line_elements.biot_savart_integral_B(r0, integration_dim="z", spatial_dim="s", r_c=r_c, dl=dl, j=j)

    np.testing.assert_allclose(B_line_el.sel(s=['x', 'z']), [0, 0])
    np.testing.assert_allclose(B_line_el.sel(s='y'), Btheta_analytic)


def test_inf_wire_line_el_vector_potential(setup):
    I, a, r0, r_c = setup
    dl = r_c.differentiate("z")  # vector
    j = dl * I  # vector with length element
    dl = (dl ** 2).sum("s") ** 0.5  # actual lengths
    j = j / dl  # normalization to line element length.

    l = 10000
    # Approximation for a << l
    # https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Book%3A_Electricity_and_Magnetism_(Tatum)/09%3A_Magnetic_Potential/9.03%3A_Long%2C_Straight%2C_Current-carrying_Conductor

    Az_analytical = (np.log(2 * l) - np.log(a)) * mu_0 * I / (2 * np.pi)

    A_calc = line_elements.biot_savart_integral_A(r0, r_c=r_c, dl=dl, j=j,
                                                  integration_dim="z",
                                                  spatial_dim="s")

    np.testing.assert_allclose(A_calc.sel(s=['x', 'y']), [0, 0])
    np.testing.assert_allclose(A_calc.sel(s='z'), Az_analytical)
