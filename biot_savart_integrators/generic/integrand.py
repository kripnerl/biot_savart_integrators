"""Numerical routines for Biot-Savart integration of a generic current density

The routines expect (dask-chunked) xarray inputs in order to facilitate
memory-efficient and parallelized computation.
"""
from scipy.constants import mu_0, pi
from .utils import cross


_SI_FACTOR = mu_0 / (4*pi)


def biot_savart_integrand(r, r_j, j, spatial_dim):
    R = r - r_j
    numerator = cross(j, R, spatial_dim)
    denominator = (R**2).sum(dim=spatial_dim)**(3/2.)
    integrand = _SI_FACTOR * numerator / denominator
    return integrand