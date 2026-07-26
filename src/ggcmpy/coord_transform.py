import ggcmpy
from ggcmpy.openggcm import _cotr_geo_sm_lat_lon, _cotr_sm_geo_lat_lon
import numpy as np
import scipy.interpolate
import xarray as xr


def transform_geo_to_sm(da: xr.DataArray) -> xr.DataArray:
    """
    Transform GEO data to SM coordinates using Inverse Lookup Bilinear Interpolation.
    Target: SM Grid. Source: GEO Grid.
    """

    lat_name = "lat" if "lat" in da.coords else "lats"
    lon_name = "lon" if "lon" in da.coords else "longs"

    if lat_name == "lats":
        return da  # Already in SM

    plot_data = da.squeeze().transpose(lat_name, lon_name)
    lat_vals = plot_data.coords[lat_name].values
    lon_vals = plot_data.coords[lon_name].values
    z_vals = plot_data.values

    time_val = np.datetime64(np.atleast_1d(plot_data.coords["time"].values)[0])

    # 1. PREPARE THE SOURCE GRID (GEO)
    # RegularGridInterpolator strictly requires monotonically ascending coordinates.
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        z_vals = z_vals[::-1, :]

    # Pad longitudes to seamlessly wrap the dateline during lookup.
    # Robust check: if the array has duplicate boundaries (e.g. 0 and 360), slice them.
    if np.isclose(lon_vals[0] + 360, lon_vals[-1]):
        lon_ext = np.concatenate([lon_vals[:-1] - 360, lon_vals, lon_vals[1:] + 360])
        z_ext = np.concatenate([z_vals[:, :-1], z_vals, z_vals[:, 1:]], axis=1)
    else:
        lon_ext = np.concatenate([lon_vals - 360, lon_vals, lon_vals + 360])
        z_ext = np.concatenate([z_vals, z_vals, z_vals], axis=1)

    # Build the Bilinear Interpolator (The "Lookup Table")
    # fill_value=None allows safe linear extrapolation for the missing 0.5 degrees at the poles!
    interp = scipy.interpolate.RegularGridInterpolator(
        (lat_vals, lon_ext), z_ext, method="linear", bounds_error=False, fill_value=None
    )

    # 2. CREATE THE TARGET GRID (SM)
    # We want a clean, standard 1-degree resolution grid.
    sm_lats = np.linspace(-90, 90, 181)
    sm_lons = np.linspace(0, 360, 361)
    sm_lon_mesh, sm_lat_mesh = np.meshgrid(sm_lons, sm_lats)

    # 3. INVERSE LOOKUP (The "Pull")
    # For every SM point, find its original Geographic coordinate.
    vec_cotr = np.vectorize(
        lambda y, x: _cotr_sm_geo_lat_lon(time_val, float(y), float(x))
    )
    geo_lat_target, geo_lon_target = vec_cotr(sm_lat_mesh, sm_lon_mesh)

    # 4. BILINEAR INTERPOLATION
    # Sample the source grid at the exact coordinates we just calculated.
    query_points = np.stack([geo_lat_target, geo_lon_target], axis=-1)
    sm_z = interp(query_points)

    # Clip the extrapolated values strictly to the original dataset's bounds.
    # This prevents linear extrapolation from pushing positive fields (like flux) below zero.
    sm_z = np.clip(sm_z, np.nanmin(z_vals), np.nanmax(z_vals))

    # Package back into Xarray
    da_sm = xr.DataArray(
        data=sm_z,
        coords={"mlat": sm_lats, "mlon": sm_lons},
        dims=["mlat", "mlon"],
        name=da.name,
        attrs=da.attrs,
    ).assign_coords(time=time_val)

    return da_sm


def transform_sm_to_geo(da: xr.DataArray) -> xr.DataArray:
    """
    Transform SM data to GEO coordinates using Inverse Lookup Bilinear Interpolation.
    """

    lat_name = "lat" if "lat" in da.coords else "lats"
    lon_name = "lon" if "lon" in da.coords else "longs"

    if lat_name == "lat":
        return da  # Already in GEO

    plot_data = da.squeeze().transpose(lat_name, lon_name)
    lat_vals = plot_data.coords[lat_name].values
    lon_vals = plot_data.coords[lon_name].values
    z_vals = plot_data.values

    time_val = np.datetime64(np.atleast_1d(plot_data.coords["time"].values)[0])

    # 1. PREPARE THE SOURCE GRID (SM)
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        z_vals = z_vals[::-1, :]

    # Robust dateline padding
    if np.isclose(lon_vals[0] + 360, lon_vals[-1]):
        lon_ext = np.concatenate([lon_vals[:-1] - 360, lon_vals, lon_vals[1:] + 360])
        z_ext = np.concatenate([z_vals[:, :-1], z_vals, z_vals[:, 1:]], axis=1)
    else:
        lon_ext = np.concatenate([lon_vals - 360, lon_vals, lon_vals + 360])
        z_ext = np.concatenate([z_vals, z_vals, z_vals], axis=1)

    # Build the interpolator with extrapolation (fill_value=None) for the poles
    interp = scipy.interpolate.RegularGridInterpolator(
        (lat_vals, lon_ext), z_ext, method="linear", bounds_error=False, fill_value=None
    )

    # 2. CREATE THE TARGET GRID (GEO)
    geo_lats = np.linspace(-90, 90, 181)
    geo_lons = np.linspace(0, 360, 361)
    geo_lon_mesh, geo_lat_mesh = np.meshgrid(geo_lons, geo_lats)

    # 3. INVERSE LOOKUP (The "Pull")
    # For every GEO point, find its original SM coordinate.
    vec_cotr = np.vectorize(
        lambda y, x: _cotr_geo_sm_lat_lon(time_val, float(y), float(x))
    )
    sm_lat_target, sm_lon_target = vec_cotr(geo_lat_mesh, geo_lon_mesh)

    # 4. BILINEAR INTERPOLATION
    query_points = np.stack([sm_lat_target, sm_lon_target], axis=-1)
    geo_z = interp(query_points)

    # Clip extrapolated values to original data bounds
    geo_z = np.clip(geo_z, np.nanmin(z_vals), np.nanmax(z_vals))

    da_geo = xr.DataArray(
        data=geo_z,
        coords={"lat": geo_lats, "lon": geo_lons},
        dims=["lat", "lon"],
        name=da.name,
        attrs=da.attrs,
    ).assign_coords(time=time_val)

    return da_geo
