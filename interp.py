import numpy as np
import pandas as pd
import polars as pl
from astropy.table import Table
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

def nn_interp_hpx(
    cat: pd.DataFrame,
    nside: int = 512,
    upper_dec_limit: float = 90,
    lower_dec_limit: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
) -> np.ndarray:
    """Optimized nearest-neighbor interpolation onto a Healpix grid."""
    pix_idx = hp.ang2pix(
        nside, cat[lon_column].values, cat[lat_column].values, lonlat=True
    )
    n_pix = hp.nside2npix(nside)
    rm_arr = np.full(n_pix, np.nan, dtype=float)
    rm_arr[pix_idx] = cat[interp_column].values

    # Fill in missing values with nearest neighbor
    nan_idx = np.isnan(rm_arr)
    nan_pix = np.arange(n_pix)[nan_idx]
    ra_nan, dec_nan = hp.pix2ang(nside, nan_pix, lonlat=True)
    nan_coords = SkyCoord(ra=ra_nan, dec=dec_nan, frame="icrs", unit="deg")
    cat_coords = SkyCoord(
        ra=cat[lon_column], dec=cat[lat_column], frame="icrs", unit="deg"
    )
    match_idx, _, _ = nan_coords.match_to_catalog_sky(cat_coords)
    rm_arr[nan_idx] = cat[interp_column].values[match_idx]

    # Apply declination limits
    _, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    rm_arr[(dec_hpx > upper_dec_limit) | (dec_hpx < lower_dec_limit)] = np.nan

    return rm_arr
def idw_interp_hpx(
    cat: pl.DataFrame,
    nside: int = 512,
    upper_dec_limit: float = 90,
    lower_dec_limit: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
    search_radius: u.Quantity = 60.0 * u.arcmin,
    inner_radius: u.Quantity = 0.0,
) -> np.ndarray:
    """Inverse-square weighted interpolation onto a Healpix grid."""

    n_pix = hp.nside2npix(nside)
    value_arr = np.full(n_pix, np.nan, dtype=float)

    # Input coordinates
    cat_coords = SkyCoord(
        ra=cat[lon_column].to_numpy(),
        dec=cat[lat_column].to_numpy(),
        frame="icrs",
        unit="deg",
    )

    # Grid coordinates
    ra_hpx, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    grid_coords = SkyCoord(ra=ra_hpx, dec=dec_hpx, frame="icrs", unit="deg")

    # Match: find all catalog points within radius
    idx_cat, idx_grid, d2d, _ = grid_coords.search_around_sky(
        searcharoundcoords=cat_coords, seplimit=search_radius
    )

    # Inverse square weights
    weights = 1 / d2d.arcminute**2
    values = cat[interp_column].to_numpy()[idx_cat]
    # Apply inner radius: set weights to zero for points within inner_radius
    if inner_radius > 0 * u.arcmin:
        inner_radius = inner_radius.to(u.arcminute).value
        weights[d2d.arcminute < inner_radius] = 0

    non_nan_idx = np.isfinite(values) & np.isfinite(weights)

    # Filter out NaNs
    values = values[non_nan_idx]
    weights = weights[non_nan_idx]
    idx_grid = idx_grid[non_nan_idx]

    # Accumulate weighted sums
    num = np.bincount(idx_grid, weights * values, minlength=n_pix)
    den = np.bincount(idx_grid, weights, minlength=n_pix)
    with np.errstate(divide="ignore", invalid="ignore"):
        value_arr = num / den

    # Apply declination limits
    value_arr[(dec_hpx > upper_dec_limit) | (dec_hpx < lower_dec_limit)] = np.nan

    return value_arr