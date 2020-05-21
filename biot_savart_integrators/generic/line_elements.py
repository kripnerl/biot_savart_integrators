"""Numerical routines for Biot-Savart integration of a generic current curve

The routines expect (dask-chunked) xarray inputs in order to facilitate
memory-efficient and parallelized computation.
"""
from typing import Union, Hashable

import xarray as xr
from .integrand import biot_savart_integrand_B as bsintegrand_B
from .integrand import biot_savart_integrand_A as bsintegrand_A


def _prepare_input_(integration_dim: Hashable, spatial_dim: Hashable,
                    r_c: xr.DataArray, dl: xr.DataArray, j):

    r_c_copy = r_c.copy().drop_vars(integration_dim, errors="ignore")

    if dl is None:
        dl = r_c_copy.differentiate(integration_dim)

    if (hasattr(j, "coords") and spatial_dim in j.coords) != (spatial_dim in dl.coords):
        j_int = j * dl
    elif (hasattr(j, "coords") and spatial_dim in j.coords) and (spatial_dim in dl.coords):
        _dl = (dl ** 2).sum(spatial_dim) ** 0.5
        j_int = j * _dl
    else:  # not in both
        dirvec = r_c_copy.differentiate(integration_dim)
        dirvec = dirvec / dl  # normalized vectors
        j_int = j * dl * dirvec

    return r_c_copy, j_int


def biot_savart_integral_B(r: xr.DataArray, integration_dim: Hashable, spatial_dim: Hashable, r_c: xr.DataArray,  # noqa
                           dl: xr.DataArray = None, j: Union[xr.DataArray, int, float] = 1) -> xr.DataArray:
    """Numerical integration of B-S law for set of line elements.

    Parameters
    ----------
    r: Positions where the magnetic field is evaluated.
    r_c: Centers of line current elements.
    dl: Length of the current element.
        If None: length elements are calculated from difference of `r_c`
        along `integration_dim`.
        If `dl` is with `spatial_dim` it may be used in determination
        of the current direction.
    j: Spatial current flowing through the element in Amps.
       If `j` is without `integration` dim it is assumed to be current
       flowing through the single wire.
       If `j` is without `spatial_dim` it is assumed to be scalar
       and the current direction is calculated from `r_c`.
    integration_dim: Dimension name over which index current
                     elements.
    spatial_dim: Name of the spatial dimension.

    Returns
    -------
    Magnetic field vector at  `r`.
    """

    r_c_copy, j_int = _prepare_input_(integration_dim, spatial_dim, r_c, dl, j)

    integrand = bsintegrand_B(r, r_c_copy, j_int, spatial_dim)

    integrand = integrand.drop_vars(integration_dim, errors="ignore")
    integral = integrand.integrate(integration_dim)
    return integral


def biot_savart_integral_A(r: xr.DataArray, integration_dim: Hashable, spatial_dim: Hashable, r_c: xr.DataArray,  # noqa
                           dl: xr.DataArray = None, j: Union[xr.DataArray, int, float] = 1) -> xr.DataArray:
    """Numerical integration of B-S law for vector potential for set of line elements.

    Parameters
    ----------
    r: Positions where the vector potential is evaluated.
    r_c: Centers of line current elements.
    dl: Length of the current element.
        If None: length elements are calculated from difference of `r_c`
        along `integration_dim`.
        If `dl` is with `spatial_dim` it may be used in determination
        of the current direction.
    j: Spatial current flowing through the element in Amps.
       If `j` is without `integration` dim it is assumed to be current
       flowing through the single wire.
       If `j` is without `spatial_dim` it is assumed to be scalar
       and the current direction is calculated from `r_c`.
    integration_dim: Dimension name over which index current
                     elements.
    spatial_dim: Name of the spatial dimension.
    Returns
    -------
    Vector potential at  `r`.
    """

    r_c_copy, j_int = _prepare_input_(integration_dim, spatial_dim, r_c, dl, j)

    integrand = bsintegrand_A(r, r_c_copy, j_int, spatial_dim)

    integrand = integrand.drop_vars(integration_dim, errors="ignore")
    integral = integrand.integrate(integration_dim)
    return integral
