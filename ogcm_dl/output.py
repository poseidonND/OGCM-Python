"""NetCDF output (ports initNetCDF_adc, initNetCDF_BCSL, UpdateNetCDF, writeBCSLNC).

Variable layout matches the Fortran:

OutType 3 / 4 (TS or TS+UV on ADCIRC grid):
  dims: time(unlimited), strlen(16), node, nele, nvertex(=3)
  vars: time, x, y, depth, element, BPGX, BPGY, SigTS, MLD, NB, NM,
        DispX, DispY (only if OutType==4)

OutType 5 (BCSL on ADCIRC grid):
  dims: time(unlimited), strlen(16), node, nele, nvertex(=3)
  vars: time, x, y, depth, element, BCSL
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # type: ignore[import-not-found]

from .calc_adcirc import StericAdcResult, TsAdcResult, UvAdcResult
from .constants import (
    DEFLATE_LEVEL,
    DFV,
    NETCDF_DTYPE,
    SIG_T0,
)
from .mesh import AdcircMesh


def _def_var_att(ds: Dataset, sname: str, nf_type, dims: tuple[str, ...],
                 lname: str, units: str,
                 fill_value: float | None = None) -> "Dataset.variable":
    var = ds.createVariable(
        sname, nf_type, dims,
        zlib=True, complevel=DEFLATE_LEVEL,
        fill_value=fill_value,
    )
    var.long_name = lname
    var.units = units
    return var


def init_netcdf_adc(path: str | Path, mesh: AdcircMesh, out_type: int) -> None:
    """Create the ADCIRC-grid output NetCDF (initNetCDF_adc).

    Used for OutType 3 (TS) and OutType 4 (TS + UV).
    """
    if out_type not in (3, 4):
        raise ValueError(f"init_netcdf_adc only handles OutType 3 or 4 (got {out_type})")

    with Dataset(str(path), mode="w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("strlen", 16)
        ds.createDimension("node", mesh.np_)
        ds.createDimension("nele", mesh.ne)
        ds.createDimension("nvertex", 3)

        # Dimension order on disk follows the Fortran writer (which uses the
        # Fortran column-major NF90 convention): time-varying fields are
        # (time, node), the time string is (time, strlen), and the element
        # table is (nvertex, nele). This makes the file byte-shape compatible
        # with the original Fortran output.
        time_var = ds.createVariable(
            "time", "S1", ("time", "strlen"),
            zlib=True, complevel=DEFLATE_LEVEL,
        )
        time_var.long_name = "UTC datetime"
        time_var.units = "YYYY-MM-DD HH:mm"

        x_var = _def_var_att(ds, "x", "f8", ("node",), "longitude", "degrees")
        y_var = _def_var_att(ds, "y", "f8", ("node",), "latitude", "degrees")
        depth_var = _def_var_att(
            ds, "depth", "f8", ("node",), "distance below geoid", "m"
        )
        ele_var = _def_var_att(
            ds, "element", "i4", ("nvertex", "nele"), "element", "nondimensional"
        )

        _def_var_att(
            ds, "BPGX", NETCDF_DTYPE, ("time", "node"),
            "east-west depth-averaged baroclinic pressure gradient", "ms^-2",
        )
        _def_var_att(
            ds, "BPGY", NETCDF_DTYPE, ("time", "node"),
            "north-south depth-averaged baroclinic pressure gradient", "ms^-2",
        )
        _def_var_att(
            ds, "SigTS", NETCDF_DTYPE, ("time", "node"),
            "surface sigmat density", "kgm^-3",
            fill_value=SIG_T0,
        )
        _def_var_att(
            ds, "MLD", NETCDF_DTYPE, ("time", "node"),
            "mixed-layer depth ratio", "[]",
            fill_value=DFV,
        )
        _def_var_att(
            ds, "NB", NETCDF_DTYPE, ("time", "node"),
            "buoyancy frequency at the seabed", "s^-1",
        )
        _def_var_att(
            ds, "NM", NETCDF_DTYPE, ("time", "node"),
            "depth-averaged buoyancy frequency", "s^-1",
        )
        if out_type == 4:
            _def_var_att(
                ds, "DispX", NETCDF_DTYPE, ("time", "node"),
                "depth-averaged x-momentum dispersion", "ms^-2",
            )
            _def_var_att(
                ds, "DispY", NETCDF_DTYPE, ("time", "node"),
                "depth-averaged y-momentum dispersion", "ms^-2",
            )

        x_var[:] = mesh.slam
        y_var[:] = mesh.sfea
        depth_var[:] = mesh.dp
        # Element table back to 1-based for output (matches the Fortran NM).
        ele_var[:, :] = (mesh.nm + 1).T.astype(np.int32)


def init_netcdf_bcsl(path: str | Path, mesh: AdcircMesh) -> None:
    """Create the BCSL output NetCDF (initNetCDF_BCSL). Used for OutType 5."""
    with Dataset(str(path), mode="w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("strlen", 16)
        ds.createDimension("node", mesh.np_)
        ds.createDimension("nele", mesh.ne)
        ds.createDimension("nvertex", 3)

        # See note in init_netcdf_adc on dimension ordering.
        time_var = ds.createVariable(
            "time", "S1", ("time", "strlen"),
            zlib=True, complevel=DEFLATE_LEVEL,
        )
        time_var.long_name = "UTC datetime"
        time_var.units = "YYYY-MM-DD HH:mm"

        x_var = _def_var_att(ds, "x", "f8", ("node",), "longitude", "degrees")
        y_var = _def_var_att(ds, "y", "f8", ("node",), "latitude", "degrees")
        depth_var = _def_var_att(
            ds, "depth", "f8", ("node",), "distance below geoid", "m"
        )
        ele_var = _def_var_att(
            ds, "element", "i4", ("nvertex", "nele"), "element", "nondimensional"
        )
        _def_var_att(
            ds, "BCSL", NETCDF_DTYPE, ("time", "node"),
            "baroclinic sea level", "m",
        )

        x_var[:] = mesh.slam
        y_var[:] = mesh.sfea
        depth_var[:] = mesh.dp
        ele_var[:, :] = (mesh.nm + 1).T.astype(np.int32)


def _write_time(ds: Dataset, time_index: int, when: datetime) -> None:
    s = when.strftime("%Y-%m-%d %H:%M").ljust(16)[:16]
    ds.variables["time"][time_index, :] = np.array(
        list(s), dtype="S1"
    )


def update_netcdf_ts(path: str | Path, time_index: int, when: datetime,
                     ts: TsAdcResult, uv: UvAdcResult | None = None) -> None:
    """Write a single time step for OutType 3 / 4."""
    with Dataset(str(path), mode="a") as ds:
        _write_time(ds, time_index, when)
        ds.variables["BPGX"][time_index, :] = ts.bpg_adc_x
        ds.variables["BPGY"][time_index, :] = ts.bpg_adc_y
        ds.variables["SigTS"][time_index, :] = ts.sigts_adc
        ds.variables["MLD"][time_index, :] = ts.mld_adc
        ds.variables["NB"][time_index, :] = ts.nb_adc
        ds.variables["NM"][time_index, :] = ts.nm_adc
        if uv is not None:
            ds.variables["DispX"][time_index, :] = uv.dispx_adc
            ds.variables["DispY"][time_index, :] = uv.dispy_adc


def update_netcdf_bcsl(path: str | Path, time_index: int, when: datetime,
                       result: StericAdcResult) -> None:
    """Write a single time step for OutType 5."""
    with Dataset(str(path), mode="a") as ds:
        _write_time(ds, time_index, when)
        ds.variables["BCSL"][time_index, :] = result.bcsl_adc
