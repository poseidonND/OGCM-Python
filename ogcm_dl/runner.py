"""Main loop (replaces PROGRAM OGCM_DL + OGCM_Run).

Runs serially: for every output time, look up the local OGCM file in
`ogcm_data.txt`, read it, compute the requested ADCIRC-grid quantities, and
append a slab to the output NetCDF.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .calc_adcirc import (
    calc_bc2d_ts_adcirc,
    calc_bc2d_uv_adcirc,
    calc_steric_adjustment,
)
from .config import Config
from .constants import BC3D_DT_HOURS
from .interp import InterpWeights, get_interpolation_weights
from .io_local import OgcmIndex, read_ogcm_index
from .mesh import AdcircMesh, calc_areas, calc_derivatives, read_f14
from .ogcm_io import Bc3dGrid, read_bc3d_netcdf  # noqa: F401
from .output import (
    init_netcdf_adc,
    init_netcdf_bcsl,
    update_netcdf_bcsl,
    update_netcdf_ts,
)

log = logging.getLogger(__name__)


def _setup_mesh(cfg: Config) -> AdcircMesh:
    log.info("Reading fort.14 file: %s", cfg.fort14)
    mesh = read_f14(cfg.fort14)
    log.info("Mesh has NP=%d nodes, NE=%d elements", mesh.np_, mesh.ne)
    log.info("Computing element areas")
    calc_areas(mesh)
    log.info("Computing element-basis derivatives")
    calc_derivatives(mesh)
    return mesh


def _get_or_compute_weights(weights_state: dict[str, object], mesh: AdcircMesh,
                            grid: Bc3dGrid) -> InterpWeights:
    """Compute (or recompute when the OGCM grid changes) the bilinear weights.

    Mirrors the FirstCall logic in `Get_LonLatDepthTime` / `Get_Interpolation_Weights`:
    when the OGCM longitude origin changes (e.g. switch from GLBv0.08 to
    GLBy0.08), the weights are recomputed; otherwise reuse them across time
    steps.
    """
    sig = (
        int(grid.bc3d_lon.size),
        int(grid.bc3d_lat.size),
        int(grid.bc3d_z.size),
        float(grid.bc3d_lon.min()),
        float(grid.bc3d_lat.min()),
    )
    if weights_state.get("sig") == sig:
        return weights_state["weights"]  # type: ignore[return-value]

    log.info(
        "Computing interpolation weights (NX=%d, NY=%d, NZ=%d)",
        sig[0], sig[1], sig[2],
    )
    weights = get_interpolation_weights(
        mesh.slam, mesh.sfea, mesh.dp,
        grid.bc3d_lon, grid.bc3d_lat, grid.bc3d_z,
    )
    weights_state["sig"] = sig
    weights_state["weights"] = weights
    return weights


def _step_ts_uv(cfg: Config, when: datetime, time_index: int,
                mesh: AdcircMesh, index: OgcmIndex,
                weights_state: dict[str, object],
                preloaded_ts: Bc3dGrid | None = None,
                preloaded_uv: Bc3dGrid | None = None) -> None:
    """Run a single time step for OutType 3 (TS) or OutType 4 (TS + UV).

    Callers may pass `preloaded_ts` / `preloaded_uv` to bypass the netCDF
    read for this step -- useful when the OGCM grid is already in memory
    (e.g. from `ogcm_dl.regrid.convert_archv_to_glby(..., return_in_memory=True)`).
    """
    if preloaded_ts is not None:
        log.info("Using preloaded TS grid (in-memory) for %s", when.isoformat(" "))
        grid_ts = preloaded_ts
    else:
        ts_path = index.file_for(when, flag=1)
        log.info("Reading TS file: %s", ts_path)
        grid_ts = read_bc3d_netcdf(ts_path, flag=3)
    weights = _get_or_compute_weights(weights_state, mesh, grid_ts)

    log.info("Computing T & S terms on ADCIRC grid")
    ts_result = calc_bc2d_ts_adcirc(grid_ts, mesh, weights)

    uv_result = None
    if cfg.out_type == 4:
        if preloaded_uv is not None:
            log.info("Using preloaded UV grid (in-memory) for %s",
                     when.isoformat(" "))
            grid_uv = preloaded_uv
        else:
            uv_path = index.file_for(when, flag=2)
            log.info("Reading UV file: %s", uv_path)
            grid_uv = read_bc3d_netcdf(uv_path, flag=4)
        # Sanity-check: the velocity grid must match the TS grid (same NX/NY/NZ
        # and lon/lat origin) for the cached weights to be valid.
        sig_uv = (
            int(grid_uv.bc3d_lon.size),
            int(grid_uv.bc3d_lat.size),
            int(grid_uv.bc3d_z.size),
            float(grid_uv.bc3d_lon.min()),
            float(grid_uv.bc3d_lat.min()),
        )
        if weights_state.get("sig") != sig_uv:
            raise RuntimeError(
                "TS and UV files at the same time stamp use different OGCM "
                "grids; this is not supported. (TS={}, UV={})".format(
                    weights_state.get("sig"), sig_uv
                )
            )
        log.info("Computing U & V terms on ADCIRC grid")
        uv_result = calc_bc2d_uv_adcirc(grid_uv, mesh, weights)

    log.info("Writing time index %d (%s)", time_index, when.isoformat(" "))
    update_netcdf_ts(cfg.bc2d_name, time_index, when, ts_result, uv_result)


def _step_bcsl(cfg: Config, when: datetime, time_index: int,
               mesh: AdcircMesh, index: OgcmIndex,
               weights_state: dict[str, object],
               preloaded_ts: Bc3dGrid | None = None) -> None:
    """Run a single time step for OutType 5 (BCSL)."""
    if preloaded_ts is not None:
        log.info("Using preloaded TS grid (in-memory) for %s", when.isoformat(" "))
        grid = preloaded_ts
    else:
        ts_path = index.file_for(when, flag=1)
        log.info("Reading TS file: %s", ts_path)
        grid = read_bc3d_netcdf(ts_path, flag=5)
    weights = _get_or_compute_weights(weights_state, mesh, grid)

    log.info("Computing baroclinic sea level on ADCIRC grid")
    result = calc_steric_adjustment(grid, mesh, weights)
    log.info("Writing time index %d (%s)", time_index, when.isoformat(" "))
    update_netcdf_bcsl(cfg.bc2d_name, time_index, when, result)


def run(cfg: Config, ogcm_data_path: str | Path = "ogcm_data.txt") -> None:
    """Top-level runner. Replaces PROGRAM OGCM_DL + subroutine OGCM_Run.

    Parameters
    ----------
    cfg
        Parsed control file.
    ogcm_data_path
        Path to the file listing local OGCM snapshots (matches `OGCMFILE` in
        the Fortran program).
    """
    if cfg.out_type not in (3, 4, 5):
        raise ValueError(
            f"OutType={cfg.out_type} is not supported. Use 3, 4, or 5."
        )

    mesh = _setup_mesh(cfg)

    log.info("Reading OGCM index: %s", ogcm_data_path)
    index = read_ogcm_index(ogcm_data_path)
    log.info(
        "OGCM index has %d snapshots from %s to %s",
        index.n_snaps,
        index.tsnap[0].isoformat(" "),
        index.tsnap[-1].isoformat(" "),
    )

    # Initialize the output NetCDF
    out_path = Path(cfg.bc2d_name)
    if out_path.exists():
        log.warning("Output file %s already exists; overwriting", out_path)
        out_path.unlink()
    if cfg.out_type in (3, 4):
        init_netcdf_adc(out_path, mesh, cfg.out_type)
    elif cfg.out_type == 5:
        init_netcdf_bcsl(out_path, mesh)

    # Time-stepping
    step = timedelta(hours=BC3D_DT_HOURS * cfg.tmult)
    cur = cfg.ts
    time_index = 0
    weights_state: dict[str, object] = {}
    while cur <= cfg.te:
        log.info("=== Time step %d: %s ===", time_index, cur.isoformat(" "))
        if cfg.out_type in (3, 4):
            _step_ts_uv(cfg, cur, time_index, mesh, index, weights_state)
        elif cfg.out_type == 5:
            _step_bcsl(cfg, cur, time_index, mesh, index, weights_state)
        cur += step
        time_index += 1

    log.info("Done. Wrote %d time steps to %s", time_index, out_path)


def compute_bc2d_from_grid(cfg: Config, when: datetime,
                           ts_grid: Bc3dGrid,
                           uv_grid: Bc3dGrid | None = None,
                           mesh: AdcircMesh | None = None
                           ) -> "dict[str, object]":
    """Run the ADCIRC-side computation for one preloaded grid and return the
    result *without* writing any NetCDF file.

    This is the low-level primitive both `run_single_step` (for the
    single-file wrapper) and the multi-file wrapper build on.

    Returns
    -------
    dict with keys:
        "when"     : datetime  (echo of the input, for convenience when this
                                is used as a worker return value)
        "out_type" : int
        "ts"       : TsAdcResult   (present for out_type 3 or 4)
        "uv"       : UvAdcResult   (present for out_type 4)
        "bcsl"     : StericAdcResult (present for out_type 5)
    """
    if cfg.out_type not in (3, 4, 5):
        raise ValueError(
            f"OutType={cfg.out_type} is not supported. Use 3, 4, or 5."
        )
    if cfg.out_type == 4 and uv_grid is None:
        raise ValueError("OutType=4 requires uv_grid to be provided.")

    if mesh is None:
        mesh = _setup_mesh(cfg)

    weights_state: dict[str, object] = {}
    weights = _get_or_compute_weights(weights_state, mesh, ts_grid)

    out: dict[str, object] = {"when": when, "out_type": cfg.out_type}
    if cfg.out_type in (3, 4):
        log.info("Computing T & S terms on ADCIRC grid")
        out["ts"] = calc_bc2d_ts_adcirc(ts_grid, mesh, weights)
        if cfg.out_type == 4:
            # Sanity: uv grid must share the TS grid geometry so the same
            # bilinear weights apply.
            sig_uv = (
                int(uv_grid.bc3d_lon.size),
                int(uv_grid.bc3d_lat.size),
                int(uv_grid.bc3d_z.size),
                float(uv_grid.bc3d_lon.min()),
                float(uv_grid.bc3d_lat.min()),
            )
            if weights_state.get("sig") != sig_uv:
                raise RuntimeError(
                    "TS and UV grids at the same time have different geometry"
                )
            log.info("Computing U & V terms on ADCIRC grid")
            out["uv"] = calc_bc2d_uv_adcirc(uv_grid, mesh, weights)
    else:
        log.info("Computing baroclinic sea level on ADCIRC grid")
        out["bcsl"] = calc_steric_adjustment(ts_grid, mesh, weights)
    return out


def run_single_step(cfg: Config, when: datetime,
                    ts_grid: Bc3dGrid,
                    uv_grid: Bc3dGrid | None = None,
                    mesh: AdcircMesh | None = None,
                    time_index: int = 0,
                    fresh_output: bool = True) -> None:
    """Run a single OGCM_DL step with a preloaded in-memory grid.

    Bypasses reading the TS/UV NetCDF files entirely. Use when the caller
    (e.g. `run_hycom_pipeline.py`) already has the regridded fields in
    memory and just needs to reuse the ADCIRC-side computation.

    Parameters
    ----------
    cfg
        Parsed control file (used only for `out_type`, `fort14`, and
        `bc2d_name` -- `ts`, `te`, `tmult`, `bc_server` are ignored).
    when
        UTC datetime for this slab in the output NetCDF.
    ts_grid, uv_grid
        Preloaded GLBy-style grids. `uv_grid` is required only for
        `cfg.out_type == 4`. For `out_type == 5` only `ts_grid` is used.
    mesh
        Pre-built ADCIRC mesh. If None, it is read from `cfg.fort14`.
    time_index
        Slab index for the output NetCDF (default 0).
    fresh_output
        If True (default), (re-)create the output NetCDF file before writing.
    """
    if cfg.out_type not in (3, 4, 5):
        raise ValueError(
            f"OutType={cfg.out_type} is not supported. Use 3, 4, or 5."
        )
    if cfg.out_type == 4 and uv_grid is None:
        raise ValueError("OutType=4 requires uv_grid to be provided.")

    if mesh is None:
        mesh = _setup_mesh(cfg)

    out_path = Path(cfg.bc2d_name)
    if fresh_output:
        if out_path.exists():
            log.warning("Output file %s already exists; overwriting", out_path)
            out_path.unlink()
        if cfg.out_type in (3, 4):
            init_netcdf_adc(out_path, mesh, cfg.out_type)
        else:
            init_netcdf_bcsl(out_path, mesh)

    weights_state: dict[str, object] = {}
    # `_step_ts_uv` / `_step_bcsl` look up an OGCM index only when they
    # actually need to open a NetCDF; with preloaded grids they never do,
    # so we can safely pass an unused (empty) index.
    dummy_index = OgcmIndex(tsnap=[when], ts_files=[Path("in-memory")],
                            uv_files=[Path("in-memory")],
                            ssh_files=[Path("in-memory")])
    log.info("=== Single step %d: %s (in-memory) ===",
             time_index, when.isoformat(" "))
    if cfg.out_type in (3, 4):
        _step_ts_uv(cfg, when, time_index, mesh, dummy_index, weights_state,
                    preloaded_ts=ts_grid, preloaded_uv=uv_grid)
    else:
        _step_bcsl(cfg, when, time_index, mesh, dummy_index, weights_state,
                   preloaded_ts=ts_grid)
    log.info("Wrote step %d to %s", time_index, out_path)
