"""Numerical routines for Biot-Savart integration of a generic current density

The routines expect (dask-chunked) xarray inputs in order to facilitate
memory-efficient and parallelized computation.
"""
from typing import Hashable

from scipy.constants import mu_0, pi
from .utils import cross
import xarray as xr


_SI_FACTOR = mu_0 / (4*pi)


def biot_savart_integrand_B(r: xr.DataArray, r_j: xr.DataArray, # noqa
                            j: xr.DataArray, spatial_dim: Hashable):
    """Integrand of Biot-Savart law for magnetic field.

    Parameters
    ----------
    r : Positions where the magnetic field is evaluated.
    r_j : Position of the current element.
    j : Current density vector (in A/m2).
    spatial_dim: Name of the spatial dimension.

    Returns
    -------
    Integrand of Biot-Savart law for magnetic field.

    Note
    ----
    Calculates
    .. math::
    \frac{\vec j(\vec r_j) \times (\vec r_j - \vec r)}{|\vec r_j - \vec r|^3}

    """
    R = r - r_j
    numerator = cross(j, R, spatial_dim)
    denominator = (R**2).sum(dim=spatial_dim) ** (3/2.)
    integrand = _SI_FACTOR * numerator / denominator
    return integrand


def biot_savart_integrand_A(r: xr.DataArray, r_j: xr.DataArray,  # noqa
                            j: xr.DataArray, spatial_dim: Hashable):
    """Integrand of Biot-Savart law for vector potential.

    Parameters
    ----------
    r : Positions where the magnetic field is evaluated.
    r_j : Position of the current element.
    j : Current density vector (in A/m2).
    spatial_dim : Name of the spatial dimension.

    Returns
    -------
    Integrand of Biot-Savart law for vector potential.

    Note
    ----
    Calculates
    .. math::
    \frac{\vec j(\vec r_j)}{|\vec r_j - \vec r|}


    """
    R = r - r_j
    numerator = j
    denominator = (R**2).sum(dim=spatial_dim) ** (1/2.)
    integrand = _SI_FACTOR * numerator / denominator
    return integrand

