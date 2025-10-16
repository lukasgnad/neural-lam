# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computes TOA incident solar radiation compatible with ERA5.

The Top-Of-the-Atmosphere (TOA) incident solar radiation is available in the
ERA5 dataset as the parameter `toa_incident_solar_radiation` (or `tisr`). This
represents the TOA solar radiation flux integrated over a period of one hour
ending at the timestamp given by the `datetime` coordinate. See
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
https://codes.ecmwf.int/grib/param-db/?id=212.
"""

from collections.abc import Callable, Sequence
from typing import Sequence, Union
import dataclasses
import functools
import cftime

import graphcast.solar_radiation as gc_sr
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xa

_TimestampLike = Union[
    pd.Timestamp,
    cftime.DatetimeNoLeap,
    np.datetime64,
    cftime.Datetime360Day,
    str,
]
_TimedeltaLike = Union[str, pd.Timedelta, np.timedelta64]


def get_tsi(
    timestamps: Sequence[_TimestampLike], tsi_data: xa.DataArray
) -> chex.Array:
    """Returns TSI values for the given timestamps.

    TSI values are interpolated from the provided yearly TSI data.

    Args:
      timestamps: Timestamps for which to compute TSI values.
      tsi_data: A DataArray with a single dimension `time` that has coordinates in
        units of years since 0000-1-1. E.g. 2023.5 corresponds to the middle of
        the year 2023.

    Returns:
      An Array containing interpolated TSI data.
    """

    first = timestamps[0]

    if isinstance(first, np.datetime64):
        first = pd.Timestamp(first)

    # --- Gregorian (pandas.Timestamp) case ---
    if isinstance(first, pd.Timestamp):
        timestamps = pd.DatetimeIndex(timestamps)
        timestamps_date = pd.DatetimeIndex(timestamps.date)
        day_fraction = (timestamps - timestamps_date) / pd.Timedelta(days=1)
        year_length = 365 + timestamps.is_leap_year
        year_fraction = (timestamps.dayofyear - 1 + day_fraction) / year_length
        fractional_year = timestamps.year + year_fraction
        return np.interp(
            fractional_year, tsi_data.coords["time"].data, tsi_data.data
        )

    # --- 360-day calendar case (cftime) ---
    elif isinstance(first, (cftime.Datetime360Day, cftime.DatetimeNoLeap)):
        fractional_year = []
        for ts in timestamps:
            day_of_year = (ts.month - 1) * 30 + ts.day  # 1..360
            # fraction of year completed
            year_frac = (day_of_year - 1) / 360.0
            fractional_year.append(ts.year + year_frac)
        fractional_year = np.array(fractional_year)
        return np.interp(
            fractional_year, tsi_data.coords["time"].data, tsi_data.data
        )

    else:
        raise TypeError(f"Unsupported timestamp type: {type(first)}")


def _get_j2000_days(timestamp: _TimestampLike) -> float:
    """Returns number of days since J2000 epoch for Gregorian or 360-day timestamps."""

    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)

    if isinstance(timestamp, pd.Timestamp):
        return timestamp.to_julian_date() - gc_sr._J2000_EPOCH

    elif isinstance(timestamp, (cftime.Datetime360Day, cftime.DatetimeNoLeap)):
        # J2000 = 2000-01-01 12:00 (midday)
        # Compute days since 2000-01-01 00:00
        days_since_2000 = (
            (timestamp.year - 2000) * 360
            + (timestamp.month - 1) * 30
            + (timestamp.day - 1)
        )
        # Add fractional part of the day
        days_since_2000 += (
            (timestamp.hour - 12) / 24.0
            + timestamp.minute / 1440.0
            + timestamp.second / 86400.0
        )
        return days_since_2000

    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")


def get_toa_incident_solar_radiation(
    timestamps: Sequence[_TimestampLike],
    latitude: chex.Array,
    longitude: chex.Array,
    tsi_data: xa.DataArray | None = None,
    integration_period: _TimedeltaLike = gc_sr._DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = gc_sr._DEFAULT_NUM_INTEGRATION_BINS,
    use_jit: bool = False,
) -> chex.Array:
    """Computes the solar radiation incident at the top of the atmosphere.

    The solar radiation is computed for each element in `timestamps` for all the
    locations on the grid determined by the `latitude` and `longitude` parameters.

    To approximate the `toa_incident_solar_radiation` (or `tisr`) parameter from
    the ERA5 dataset, set `integration_period` to one hour (default). See
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
    https://codes.ecmwf.int/grib/param-db/?id=212.

    Args:
      timestamps: Timestamps for which to compute the solar radiation.
      latitude: The latitude coordinates in degrees of the grid for which to
        compute the solar radiation.
      longitude: The longitude coordinates in degrees of the grid for which to
        compute the solar radiation.
      tsi_data: A DataArray containing yearly TSI data as returned by a
        `TsiDataLoader`. The default is to use ERA5 compatible TSI data.
      integration_period: Timedelta to use to integrate the radiation, e.g. if
        producing radiation for 1989-11-08 21:00:00, and `integration_period` is
        "1h", radiation will be integrated from 1989-11-08 20:00:00 to 1989-11-08
        21:00:00. The default value ("1h") matches ERA5.
      num_integration_bins: Number of equally spaced bins to divide the
        `integration_period` in when approximating the integral using the
        trapezoidal rule. Performance and peak memory usage are affected by this
        value. The default (360) provides a good approximation, but lower values
        may work to improve performance and reduce memory usage.
      use_jit: Set to True to use the jitted implementation, or False (default) to
        use the non-jitted one.

    Returns:
      An 3D array with dimensions (time, lat, lon) containing the total
      top of atmosphere solar radiation integrated for the `integration_period`
      up to each timestamp.
    """
    # Add a trailing dimension to latitude to get dimensions (lat, lon).
    lat = jnp.radians(latitude).reshape((-1, 1))
    lon = jnp.radians(longitude)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)

    integration_period = pd.Timedelta(integration_period)

    if tsi_data is None:
        tsi_data = gc_sr._DEFAULT_TSI_DATA_LOADER()

    tsi = get_tsi(timestamps, tsi_data)
    fn = (
        gc_sr._get_integrated_radiation_jitted
        if use_jit
        else gc_sr._get_integrated_radiation
    )

    # Compute integral for each timestamp individually. Although this could be
    # done in one step, peak memory usage would be proportional to
    # `len(timestamps) * num_integration_bins`. Computing each timestamp
    # individually reduces this to `max(len(timestamps), num_integration_bins)`.
    # E.g. memory usage for a single timestamp, with a full 0.25° grid and 360
    # integration bins is about 1.5 GB (1440 * 721 * 361 * 4 bytes); computing
    # forcings for 40 prediction steps would require 60 GB.

    results = []
    for idx, timestamp in enumerate(timestamps):
        j2000_days = _get_j2000_days(timestamp)
        results.append(
            fn(
                j2000_days=jnp.array(j2000_days),
                sin_latitude=sin_lat,
                cos_latitude=cos_lat,
                longitude=lon,
                tsi=tsi[idx],
                integration_period=integration_period,
                num_integration_bins=num_integration_bins,
            )
        )

    return jnp.stack(results, axis=0)


def get_toa_incident_solar_radiation_for_xarray(
    data_array_like: xa.DataArray | xa.Dataset,
    tsi_data: xa.DataArray | None = None,
    integration_period: _TimedeltaLike = gc_sr._DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = gc_sr._DEFAULT_NUM_INTEGRATION_BINS,
    use_jit: bool = False,
) -> xa.DataArray:
    """Computes the solar radiation incident at the top of the atmosphere.

    This method is a wrapper for `get_toa_incident_solar_radiation` using
    coordinates from an Xarray and returning an Xarray.

    Args:
      data_array_like: A xa.Dataset or xa.DataArray from which to take the time
        and spatial coordinates for which to compute the solar radiation. It must
        contain `lat` and `lon` spatial dimensions with corresponding coordinates.
        If a `time` dimension is present, the `datetime` coordinate should be a
        vector associated with that dimension containing timestamps for which to
        compute the solar radiation. Otherwise, the `datetime` coordinate should
        be a scalar representing the timestamp for which to compute the solar
        radiation.
      tsi_data: A DataArray containing yearly TSI data as returned by a
        `TsiDataLoader`. The default is to use ERA5 compatible TSI data.
      integration_period: Timedelta to use to integrate the radiation, e.g. if
        producing radiation for 1989-11-08 21:00:00, and `integration_period` is
        "1h", radiation will be integrated from 1989-11-08 20:00:00 to 1989-11-08
        21:00:00. The default value ("1h") matches ERA5.
      num_integration_bins: Number of equally spaced bins to divide the
        `integration_period` in when approximating the integral using the
        trapezoidal rule. Performance and peak memory usage are affected by this
        value. The default (360) provides a good approximation, but lower values
        may work to improve performance and reduce memory usage.
      use_jit: Set to True to use the jitted implementation, or False to use the
        non-jitted one.

    Returns:
      xa.DataArray with dimensions `(time, lat, lon)` if `data_array_like` had
      a `time` dimension; or dimensions `(lat, lon)` otherwise. The `datetime`
      coordinates and those for the dimensions are copied to the returned array.
      The array contains the total top of atmosphere solar radiation integrated
      for `integration_period` up to the corresponding `datetime`.

    Raises:
      ValueError: If there are missing coordinates or dimensions.
    """
    missing_dims = set(["lat", "lon"]) - set(data_array_like.dims)
    if missing_dims:
        raise ValueError(
            f"'{missing_dims}' dimensions are missing in `data_array_like`."
        )

    missing_coords = set(["datetime", "lat", "lon"]) - set(
        data_array_like.coords
    )
    if missing_coords:
        raise ValueError(
            f"'{missing_coords}' coordinates are missing in `data_array_like`."
        )

    if "time" in data_array_like.dims:
        timestamps = data_array_like.coords["datetime"].data
    else:
        timestamps = [data_array_like.coords["datetime"].data.item()]

    radiation = get_toa_incident_solar_radiation(
        timestamps=timestamps,
        latitude=data_array_like.coords["lat"].data,
        longitude=data_array_like.coords["lon"].data,
        tsi_data=tsi_data,
        integration_period=integration_period,
        num_integration_bins=num_integration_bins,
        use_jit=use_jit,
    )

    if "time" in data_array_like.dims:
        output = xa.DataArray(radiation, dims=("time", "lat", "lon"))
    else:
        output = xa.DataArray(radiation[0], dims=("lat", "lon"))

    # Preserve as many of the original coordinates as possible, so long as the
    # dimension or the coordinate still exist in the output array.
    for k, coord in data_array_like.coords.items():
        if set(coord.dims).issubset(set(output.dims)):
            output.coords[k] = coord
    return output
