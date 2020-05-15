"""Numerical routines for Biot-Savart integration of a generic current curve

The routines expect (dask-chunked) xarray inputs in order to facilitate
memory-efficient and parallelized computation.
"""
import xarray as xr
from .integrand import biot_savart_integrand_B as bsintegrand_B
from .integrand import biot_savart_integrand_A as bsintegrand_A


def biot_savart_integral(r: xr.DataArray, r_c: xr.DataArray,
                         dl: xr.DataArray, j: xr.DataArray,
                         integration_dim: str, spatial_dim: str) -> xr.DataArray:
    """Numerical integration of B-S law for set of line elements.

    Parameters
    ----------
    r: Positions where the magnetic field is evaluated.
    r_c: Centers of line current elements.
    dl: Length of the current element (scalar).
    j: Spatial current flowing through the element in Amps.
    integration_dim: Dimension name over which index current
                     elements.
    spatial_dim: Name of the spatial dimension.

    Returns
    -------
    Magnetic field vector at  `r`.
    """

    j_int = j * dl
    integrand = bsintegrand_B(r, r_c, j_int, spatial_dim)

    # TODO: Trapez rule used by integrate is more precise method then simple
    # summation. However, this method may brake if the coordinates are assign
    # to the integration_dim
    # integral = integrand.sum(integration_dim)
    integral = integrand.integrate(integration_dim)
    return integral


def biot_savart_potential_integral(r: xr.DataArray, r_c: xr.DataArray,
                         dl: xr.DataArray, j: xr.DataArray,
                         integration_dim: str, spatial_dim: str) -> xr.DataArray:
    """Numerical integration of B-S law for vector potential for set of line elements.

    Parameters
    ----------
    r - Positions where the magnetic field is evaluated.
    r_c - Centers of line current elements.
    dl - Length of the current element (scalar).
    j - Spatial current flowing through the element in Amps.
    integration_dim - Dimension name over which index current
                            elements.
    spatial_dim - Name of the spatial dimension.

    Returns
    -------
    Vector potential at  `r`.
    """
    j_int = j * dl
    integrand = bsintegrand_A(r, r_c, j_int, spatial_dim)

    # TODO: Trapez rule used by integrate is more precise method then simple
    # summation. However, this method may brake if the coordinates are assign
    # to the integration_dim
    # integral = integrand.sum(integration_dim)
    integral = integrand.integrate(integration_dim)
    return integral
