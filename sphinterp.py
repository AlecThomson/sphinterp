"""Spherical interpolation functions for HEALPix grids."""
from collections.abc import Callable
from typing import Literal, NamedTuple, TypeAlias

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import polars as pl
from astropy.coordinates import SkyCoord
from astropy.table import Table


class ColArrays(NamedTuple):
    """Named tuple for holding arrays of longitude, latitude, and values."""

    lon_array: np.typing.NDArray[np.floating]
    lat_array: np.typing.NDArray[np.floating]
    val_array: np.typing.NDArray[np.floating]


def _cols_to_arrays(
    cat: pd.DataFrame | Table | pl.DataFrame,
    interp_column: str,
    lon_column: str,
    lat_column: str,
) -> ColArrays:
    if isinstance(cat, Table):
        return ColArrays(
            lon_array=cat[lon_column].data,
            lat_array=cat[lat_column].data,
            val_array=cat[interp_column].data,
        )
    return ColArrays(
        lon_array=cat[lon_column].to_numpy(),
        lat_array=cat[lat_column].to_numpy(),
        val_array=cat[interp_column].to_numpy(),
    )


def nn_interp_hpx(
    cat: pd.DataFrame | Table | pl.DataFrame,
    nside: int = 512,
    upper_lat_limit_deg: float = 90,
    lower_lat_limit_deg: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
    frame: str = "icrs",
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
        frame (str, optional): Coordinate frame (e.g. `icrs`, `fk5`, `galactic` etc). Defaults to "icrs".

    Returns:
        np.typing.NDArray[np.floating]: Values interpolated onto HEALPix grid

    """
    col_arrays = _cols_to_arrays(
        cat=cat,
        lon_column=lon_column,
        lat_column=lat_column,
        interp_column=interp_column,
    )

    n_pix = hp.nside2npix(nside)
    interp_arr = np.full(n_pix, np.nan, dtype=float)

    # Fill in missing values with nearest neighbour
    nan_idx = np.isnan(interp_arr)
    nan_pix = np.arange(n_pix)[nan_idx]
    ra_nan, dec_nan = hp.pix2ang(nside, nan_pix, lonlat=True)
    nan_coords = SkyCoord(ra=ra_nan, dec=dec_nan, frame="icrs", unit="deg")
    cat_coords = SkyCoord(
        col_arrays.lon_array,
        col_arrays.lat_array,
        frame=frame,
        unit="deg",
    )
    match_idx, _, _ = nan_coords.match_to_catalog_sky(cat_coords)
    interp_arr[nan_idx] = col_arrays.val_array[match_idx]
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
    search_radius: u.Quantity = 60.0 * u.arcmin,
    inner_radius: u.Quantity = 0.0 * u.deg,
) -> np.typing.NDArray[np.floating]:
    """Inverse-square weighted interpolation onto a HEALPix grid.

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
        inner_radius (u.Quantity, optional): Inner radius cut in interpolation. Defaults to 0*u.deg.

    Returns:
        np.typing.NDArray[np.floating]: Values interpolated onto HEALPix grid

    """
    n_pix = hp.nside2npix(nside)
    value_arr = np.full(n_pix, np.nan, dtype=float)

    col_arrays = _cols_to_arrays(
        cat=cat,
        lon_column=lon_column,
        lat_column=lat_column,
        interp_column=interp_column,
    )

    cat_coords = SkyCoord(
        col_arrays.lon_array,
        col_arrays.lat_array,
        frame=frame,
        unit="deg",
    )

    # Grid coordinates
    ra_hpx, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    grid_coords = SkyCoord(ra=ra_hpx, dec=dec_hpx, frame="icrs", unit="deg")
    idx_cat, idx_grid, d2d, _ = grid_coords.search_around_sky(
        searcharoundcoords=cat_coords,
        seplimit=search_radius,
    )

    # Inverse square weights
    weights = 1 / d2d.arcminute**2
    values = col_arrays.val_array[idx_cat]
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
    value_arr[(dec_hpx > upper_lat_limit_deg) | (dec_hpx < lower_lat_limit_deg)] = (
        np.nan
    )

    return value_arr


def _resolve_combine_func(  # noqa: PLR0911
    combine_func: Literal["mean", "median", "min", "max", "sum", "std", "count"]
    | Callable[[pl.Expr], pl.Expr],
) -> Callable[[pl.Expr], pl.Expr]:
    if callable(combine_func):
        return combine_func

    match combine_func:
        case "mean":
            return pl.Expr.mean
        case "median":
            return pl.Expr.median
        case "min":
            return pl.Expr.min
        case "max":
            return pl.Expr.max
        case "sum":
            return pl.Expr.sum
        case "std":
            return pl.Expr.std
        case "count":
            return pl.Expr.count
        case _:
            msg = (
                f"Unknown combine_func '{combine_func}'. Choose from 'mean', "
                "'median', 'min', 'max', 'sum', 'std', 'count', or pass a "
                "callable that maps a polars expression to an aggregated expression."
            )
            raise ValueError(msg)


CombineName: TypeAlias = Literal["mean", "median", "min", "max", "sum", "std", "count"]


def bin_interp_hpx(
    cat: pd.DataFrame | Table | pl.DataFrame,
    nside: int = 512,
    upper_lat_limit_deg: float = 90,
    lower_lat_limit_deg: float = -90,
    interp_column: str = "rm",
    lon_column: str = "ra",
    lat_column: str = "dec",
    frame: str = "icrs",
    combine_func: CombineName | Callable[[pl.Expr], pl.Expr] = "mean",
) -> np.typing.NDArray[np.floating]:
    """Fast binning of a catalogue onto a HEALPix grid using a combine function.

    Sources falling in the same HEALPix pixel are combined with `combine_func`.
    Pixels containing no sources are NaN.

    Example usage:
    >>> my_cat = Table.read("cat.fits") # read data
    >>> interp_array = bin_interp_hpx(cat=my_cat) # pixel mean
    >>> median_array = bin_interp_hpx(cat=my_cat, combine_func="median") # pixel median
    >>> mad_array = bin_interp_hpx(
    ...     cat=my_cat,
    ...     combine_func=lambda x: (x - x.median()).abs().median(),
    ... ) # custom polars expression
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
        combine_func (CombineName | Callable[[pl.Expr], pl.Expr], optional): How to combine
            sources within a pixel. Either one of "mean", "median", "min", "max", "sum",
            "std" or "count", or a callable that takes a polars expression and returns an
            aggregated expression (e.g. `pl.Expr.mean`). Defaults to "mean".

    Returns:
        np.typing.NDArray[np.floating]: Values binned onto HEALPix grid

    Raises:
        ValueError: If `combine_func` is a string that is not a known option.

    """
    agg_func = _resolve_combine_func(combine_func)

    col_arrays = _cols_to_arrays(
        cat=cat,
        lon_column=lon_column,
        lat_column=lat_column,
        interp_column=interp_column,
    )
    lon, lat, val = (
        np.asarray(col_arrays.lon_array, dtype=float),
        np.asarray(col_arrays.lat_array, dtype=float),
        np.asarray(col_arrays.val_array, dtype=float),
    )

    # The HEALPix grid is defined in ICRS, so convert the catalogue if needed
    if frame != "icrs":
        icrs_coords = SkyCoord(lon, lat, frame=frame, unit="deg").icrs
        lon, lat = icrs_coords.ra.deg, icrs_coords.dec.deg

    # Drop non-finite entries
    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)

    n_pix = hp.nside2npix(nside)
    value_arr = np.full(n_pix, np.nan, dtype=float)

    pix_idx = hp.ang2pix(nside, lon[good], lat[good], lonlat=True)
    grouped = (
        pl.DataFrame({"pix_idx": pix_idx, "value": val[good]})
        .group_by("pix_idx")
        .agg(agg_func(pl.col("value")).alias("value"))
    )
    value_arr[grouped["pix_idx"].to_numpy()] = grouped["value"].to_numpy()

    # Apply declination limits
    _, dec_hpx = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    value_arr[(dec_hpx > upper_lat_limit_deg) | (dec_hpx < lower_lat_limit_deg)] = (
        np.nan
    )

    return value_arr
