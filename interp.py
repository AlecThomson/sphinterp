import numpy as np
import pandas as pd
import polars as pl
from astropy.table import Table
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u


def nn_interp_hpx(
    cat: pd.DataFrame | Table | pl.DataFrame,
    nside: int = 512,
    upper_lat_limit_deg: float = 90,
    lower_lat_limit_deg: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
) -> np.typing.NDArray[np.floating]:
    """Fast nearest-neighbor interpolation onto a HEALPix grid.

    Example usage:
        >>> my_cat = Table.read("cat.fits") # read data
        >>> interp_array = nn_interp_hpx(cat=my_cat) # interpolate
        >>> _ = hp.mollview(interp_array) # plot with healpy

    Args:
        cat (pd.DataFrame | Table | pl.DataFrame): Table-like catalogue dataframe
        nside (int, optional): HEALPix Nside. Defaults to 512.
        upper_lat_limit_deg (float, optional): Upper latitude limit in degreees. Defaults to 90.
        lower_lat_limit_deg (float, optional): Lower latitude limit in degreees. Defaults to -90.
        interp_column (str, optional): Column to interpolate. Defaults to "rm".
        lon_column (str, optional): Column containing longitude in degrees. Defaults to "ra".
        lat_column (str, optional): Column containing latitude. Defaults to "dec".

    Returns:
        np.typing.NDArray[np.floating]: Values interpolated onto HEALPix grid
    """

    lon_array = np.array(cat[lon_column])
    lat_array = np.array(cat[lat_column])

    pix_idx = hp.ang2pix(nside, lon_array, lat_array, lonlat=True)
    n_pix = hp.nside2npix(nside)
    interp_arr = np.full(n_pix, np.nan, dtype=float)
    value_arr = np.array(cat[interp_column])
    interp_arr[pix_idx] = value_arr

    # Fill in missing values with nearest neighbour
    nan_idx = np.isnan(interp_arr)
    nan_pix = np.arange(n_pix)[nan_idx]
    ra_nan, dec_nan = hp.pix2ang(nside, nan_pix, lonlat=True)
    nan_coords = SkyCoord(ra=ra_nan, dec=dec_nan, frame="icrs", unit="deg")
    cat_coords = SkyCoord(
        ra=cat[lon_column], dec=cat[lat_column], frame="icrs", unit="deg"
    )
    match_idx, _, _ = nan_coords.match_to_catalog_sky(cat_coords)
    interp_arr[nan_idx] = interp_arr[match_idx]
    _, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    interp_arr[(dec_hpx > upper_lat_limit_deg) | (dec_hpx < lower_lat_limit_deg)] = (
        np.nan
    )

    return interp_arr


def idw_interp_hpx(
    cat: pd.DataFrame | Table | pl.DataFrame,
    nside: int = 512,
    upper_lat_limit_deg: float = 90,
    lower_lat_limit_deg: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
    frame: str = "icrs",
    search_radius: u.Quantity = 60.0 * u.arcmin,  # type: ignore
    inner_radius: u.Quantity = 0.0 * u.deg,  # type: ignore
) -> np.typing.NDArray[np.floating]:
    """Inverse-square weighted interpolation onto a Healpix grid.

    Example usage:
        >>> my_cat = Table.read("cat.fits") # read data
        >>> interp_array = idw_interp_hpx(cat=my_cat) # interpolate
        >>> _ = hp.mollview(interp_array) # plot with healpy

    Args:
        cat (pd.DataFrame | Table | pl.DataFrame): Table-like catalogue dataframe
        nside (int, optional): HEALPix Nside. Defaults to 512.
        upper_lat_limit_deg (float, optional): Upper latitude limit in degreees. Defaults to 90.
        lower_lat_limit_deg (float, optional): Lower latitude limit in degreees. Defaults to -90.
        interp_column (str, optional): Column to interpolate. Defaults to "rm".
        lon_column (str, optional): Column containing longitude in degrees. Defaults to "ra".
        lat_column (str, optional): Column containing latitude. Defaults to "dec".
        frame (str, optional): Coordinate frame (e.g. `icrs`, `fk5`, `galactic` etc). Defaults to "icrs".
        search_radius (u.Quantity, optional): Outer radius to interpolate. Larger values will be smoother, but slower. Defaults to 60.0*u.arcmin.
        search_radius (u.Quantity, optional): Inner radius cut in interpolation. Defaults to 0*u.deg.

    Returns:
        np.typing.NDArray[np.floating]: Values interpolated onto HEALPix grid
    """

    n_pix = hp.nside2npix(nside)
    value_arr = np.full(n_pix, np.nan, dtype=float)

    lon_array = np.array(cat[lon_column])
    lat_array = np.array(cat[lat_column])
    value_array = np.array(cat[interp_column])

    cat_coords = SkyCoord(
        lon_array,
        lat_array,
        frame=frame,
        unit="deg",
    )

    # Grid coordinates
    ra_hpx, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    grid_coords = SkyCoord(ra=ra_hpx, dec=dec_hpx, frame="icrs", unit="deg")
    idx_cat, idx_grid, d2d, _ = grid_coords.search_around_sky(
        searcharoundcoords=cat_coords, seplimit=search_radius
    )

    # Inverse square weights
    weights = 1 / d2d.arcminute**2
    values = value_array[idx_cat]
    # Apply inner radius: set weights to zero for points within inner_radius
    if inner_radius > 0 * u.arcmin:  # type: ignore
        inner_radius = inner_radius.to(u.arcminute).value  # type: ignore
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
    value_arr[(dec_hpx > upper_lat_limit_deg) | (dec_hpx < lower_lat_limit_deg)] = (
        np.nan
    )

    return value_arr
