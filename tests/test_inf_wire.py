from biot_savart_integrators.generic import curve as generic_curve
from biot_savart_integrators.generic import line_elements

import numpy as np
import xarray as xr
from scipy.constants import pi, mu_0

import pytest


@pytest.fixture
def setup():
    I = 1.15766
    a = 1.23
    # high phi resolution to fulfill tolerance

    zs = xr.DataArray(np.linspace(-10000, 10000, int(1e6)), dims=['z'], name='z')
    r0 = xr.DataArray([a, 0, 2.1], coords=[('s', list('xyz'))], name='r0')
    r_c = xr.concat([xr.zeros_like(zs), xr.zeros_like(zs), zs], r0.s)
    return I, a, r0, r_c


def test_inf_wire_line_el(setup):
    I, a, r0, r_c = setup
    dl = r_c.differentiate("z")  # vector
    j = dl * I  # vector with length element
    dl = (dl ** 2).sum("s") ** 0.5  # actual lengths
    j = j / dl  # normalization to line element length.

    Btheta_analytic = mu_0 * I / (2 * np.pi * a)

    B_line_el = line_elements.biot_savart_integral(r0, r_c, dl, j,
                                                   integration_dim="z",
                                                   spatial_dim="s")

    np.testing.assert_allclose(B_line_el.sel(s=['x', 'z']), [0, 0])
    np.testing.assert_allclose(B_line_el.sel(s='y'), Btheta_analytic)
