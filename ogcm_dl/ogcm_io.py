"""HYCOM/GOFS NetCDF reader.

Ports `Read_BC3D_NetCDF`, `Get_LonLatDepthTime`, and `read_nc_var` from the
Fortran program. Returns the data already unscaled (scale_factor + add_offset
applied), with the fill mask preserved as `FV` (-3e4) so that downstream
calculations behave identically to the Fortran version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # type: ignore[import-not-found]

from .constants import FV, FVP

# Variable name mapping per `flag` (matches Read_BC3D_NetCDF in the Fortran):
#   1 -> Salinity, Temperature
#   2 -> u-velocity, v-velocity
#   3 -> Salinity, Temperature (same as 1; Fortran uses this for ADCIRC TS)
#   4 -> u-velocity, v-velocity (same as 2; for ADCIRC UV)
#   5 -> Salinity, Temperature (used for Calc_Steric_Adjustment)
_VAR_PAIRS = {
    1: ("salinity", "water_temp"),
    2: ("water_u", "water_v"),
    3: ("salinity", "water_temp"),
    4: ("water_u", "water_v"),
    5: ("salinity", "water_temp"),
}


@dataclass
class Bc3dGrid:
    """Loaded HYCOM/GOFS grid + 3D fields.

    Attributes
    ----------
    bc3d_lon : np.ndarray, shape (nx,)
    bc3d_lat : np.ndarray, shape (ny,)
    bc3d_z : np.ndarray, shape (nz,)
        Depth levels [m, positive down].
    bc3d_sp : np.ndarray, shape (nz, ny, nx)
        First field: salinity (flag=1, 3, 5) or u-velocity (flag=2, 4).
        Cells with the original `_FillValue` are stored as FV.
    bc3d_t : np.ndarray, shape (nz, ny, nx)
        Second field: temperature or v-velocity.
    """

    bc3d_lon: np.ndarray
    bc3d_lat: np.ndarray
    bc3d_z: np.ndarray
    bc3d_sp: np.ndarray
    bc3d_t: np.ndarray


def _read_var(nc: Dataset, name: str) -> np.ndarray:
    """Read a 3D scaled field, replacing fill values with FV (matches Fortran).

    The Fortran uses raw shorts plus scale_factor / add_offset / _FillValue.
    netCDF4 normally auto-applies scaling for us, but the auto-mask hides the
    distinction we need. We therefore disable masking, replicate the same
    `Var > FV+1e-3 -> Var*SF+OS` rule, and leave fills as the original FV.
    """
    var = nc.variables[name]
    var.set_auto_mask(False)
    var.set_auto_scale(False)

    raw = np.asarray(var[:], dtype=np.float64)
    # The Fortran reads shape (NX, NY, NZ); netCDF on disk stores as
    # (time, depth, lat, lon). Strip the (typically singleton) time dim.
    if raw.ndim == 4:
        if raw.shape[0] != 1:
            raise ValueError(
                f"Expected singleton time dimension for {name}, got "
                f"shape {raw.shape}."
            )
        raw = raw[0]
    if raw.ndim != 3:
        raise ValueError(
            f"Variable {name!r} has unexpected ndim={raw.ndim} (shape={raw.shape})."
        )

    fv_attr = float(getattr(var, "_FillValue"))
    sf_attr = float(getattr(var, "scale_factor"))
    os_attr = float(getattr(var, "add_offset"))

    # Threshold = FV + 1e-3 (Fortran adds a tiny buffer); above this we apply
    # the unscaling, below it (== fill) we mark with our internal FV sentinel.
    fv_thresh = fv_attr + 1.0e-3
    out = np.where(raw > fv_thresh, raw * sf_attr + os_attr, FV).astype(np.float64)
    return out


def read_bc3d_netcdf(path: str | Path, flag: int) -> Bc3dGrid:
    """Read a HYCOM/GOFS NetCDF snapshot (Read_BC3D_NetCDF + Get_LonLatDepthTime).

    `flag` mirrors the Fortran flag:
      1 / 3 / 5 -> salinity + water_temp
      2 / 4     -> water_u + water_v

    Returns a `Bc3dGrid` with fields shaped (nz, ny, nx). Fill values are
    represented by the constant `FV`.
    """
    if flag not in _VAR_PAIRS:
        raise ValueError(f"Unknown flag={flag} for read_bc3d_netcdf")
    var_a, var_b = _VAR_PAIRS[flag]

    with Dataset(str(path), mode="r") as nc:
        bc3d_lat = np.asarray(nc.variables["lat"][:], dtype=np.float64)
        bc3d_lon = np.asarray(nc.variables["lon"][:], dtype=np.float64)
        bc3d_z = np.asarray(nc.variables["depth"][:], dtype=np.float64)
        bc3d_sp = _read_var(nc, var_a)
        bc3d_t = _read_var(nc, var_b)

    return Bc3dGrid(
        bc3d_lon=bc3d_lon,
        bc3d_lat=bc3d_lat,
        bc3d_z=bc3d_z,
        bc3d_sp=bc3d_sp,
        bc3d_t=bc3d_t,
    )


def is_valid_corner(values: np.ndarray) -> np.ndarray:
    """Boolean mask for cells whose value is above the fill threshold."""
    return values > FVP
