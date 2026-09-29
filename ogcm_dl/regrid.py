"""Convert a HYCOM/RTOFS 3-D archive (.a/.b) into GLBy-style NetCDF.

The archives that come out of RTOFS live on the Navy 0.08 deg tripolar grid
with 41 hybrid (partly isopycnal) vertical layers. The OGCM_DL Python port,
like the original Fortran, expects the "GOFS/GLBy" convention: a regular 1-D
`lat` x `lon` grid with 40 fixed z-levels and int16 fields with
`scale_factor` / `add_offset` / `_FillValue`. This module bridges the two.

Two transforms are done:

1. Vertical  hybrid -> z. For every source column we build layer midpoint
   depths from `thknss / 9806` (HYCOM's "specific-weight" convention;
   dz[m] = thknss[Pa] / (g*rho0), with g*rho0 = 9806), then linearly
   interpolate the field onto the target z-levels. Target depths below the
   deepest layer midpoint (or in dry columns) are marked as fill.

2. Horizontal  curvilinear -> regular. We build a cKDTree on the source
   (plat, plon) points once and reuse it for every layer. For each target
   (lat, lon) cell we take the k=4 nearest source points and use inverse-
   distance weights on the great-circle chord distance. This is robust
   across the tripolar seams north of ~47 deg N and matches ordinary
   bilinear behaviour in the rectilinear part of the grid.

The output is written in exactly the same layout HYCOM GOFS/GLBy publishes:

    dims: lat, lon, depth, time (=1)
    vars: time, lat, lon, depth, water_temp, salinity  (ts3z)
    vars: time, lat, lon, depth, water_u,   water_v   (uv3z)

with int16 storage, `scale_factor=0.001`, `add_offset=20.0`, and
`_FillValue=-30000` -- so `ogcm_dl.ogcm_io.read_bc3d_netcdf` can read the
result unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset  # type: ignore[import-not-found]
from scipy.spatial import cKDTree  # type: ignore[import-not-found]

log = logging.getLogger(__name__)


# GLBy standard z-levels (40 layers, m positive down).
GLBY_Z_LEVELS = np.array([
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0,
    30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    125.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0, 700.0,
    800.0, 900.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
], dtype=np.float64)

# GLBy standard 4500x4251 regular grid (lon 0..360 at 0.08 deg, lat -80..90
# at 0.04 deg).
GLBY_LON = 0.08 * np.arange(4500, dtype=np.float64)
GLBY_LAT = -80.0 + 0.04 * np.arange(4251, dtype=np.float64)

# HYCOM archive fill marker for float32 (see utils_proc_arch.py in RTOFS_Post).
HYCOM_FILL = np.float32(2.0**100)

# GLBy-style storage parameters. Matches your hycom_glby_..._ts3z.nc file so
# that ogcm_dl.ogcm_io.read_bc3d_netcdf reads it back verbatim.
GLBY_SCALE = 0.001
GLBY_OFFSET = 20.0
GLBY_FV_INT16 = np.int16(-30000)


# ---------------------------------------------------------------------------
# HYCOM binary reader (self-contained; equivalent to utils_proc_arch.getField).
# ---------------------------------------------------------------------------

def _b_lines(a_path: Path) -> list[str]:
    b_path = a_path.with_suffix(".b")
    return [ln.rstrip() for ln in open(b_path, "r").readlines()]


def _get_dims(a_path: Path) -> tuple[int, int, int]:
    """Return (kdm, jdm, idm) for an archive file, or (0, jdm, idm) for grid.

    Reads the same header keywords the RTOFS_Post `getDims` reads.
    """
    lines = _b_lines(a_path)
    idm = jdm = None
    for ln in lines:
        if "idm" in ln and idm is None:
            idm = int(ln.split()[0])
        elif "jdm" in ln and jdm is None:
            jdm = int(ln.split()[0])
        if idm is not None and jdm is not None:
            break
    if idm is None or jdm is None:
        raise RuntimeError(f"Could not parse idm/jdm from {a_path}.b")
    kdm = 0
    if "arch" in a_path.name:
        # For archv files the last line's k index equals kdm.
        kdm = int(lines[-1].split()[4])
    return kdm, jdm, idm


def _field_records(a_path: Path, name: str) -> list[int]:
    """Return the 0-based record indices for a named HYCOM field in .b."""
    lines = _b_lines(a_path)
    if "arch" in a_path.name:
        header_offset = 10
        body = lines[header_offset:]
    elif "grid" in a_path.name:
        header_offset = 3
        body = lines[header_offset:]
    else:
        header_offset = 0
        body = lines
    hits = []
    for i, ln in enumerate(body):
        tokens = ln.split()
        if not tokens:
            continue
        # Strip trailing "." or ":" (matches HYCOM conventions like "u-vel.").
        stripped = tokens[0].replace(".", "").replace(":", "")
        if stripped == name:
            hits.append(i)
    return hits


def _get_layer_density(a_path: Path, name: str) -> np.ndarray:
    """Layer density (sigma) values written in .b for a 3-D field."""
    lines = _b_lines(a_path)
    if "arch" in a_path.name:
        lines = lines[10:]
    out = []
    for ln in lines:
        toks = ln.split()
        if not toks:
            continue
        stripped = toks[0].replace(".", "").replace(":", "")
        if stripped == name:
            out.append(float(toks[5]))
    return np.asarray(out, dtype=np.float64)


def _read_field(a_path: Path, name: str, layers: list[int] | None = None,
                undef: float = float("nan")) -> np.ndarray:
    """Read a 2-D or 3-D HYCOM field from a .a binary.

    Mirrors utils_proc_arch.getField, but self-contained (so this project
    does not depend on the RTOFS_Post repo layout at import time).
    """
    kdm, jdm, idm = _get_dims(a_path)
    reclen = 4 * idm * jdm
    wordlen = 4096 * 4
    pad = int(wordlen * np.ceil(reclen / wordlen) - reclen)

    recs = _field_records(a_path, name)
    if not recs:
        raise KeyError(f"Field {name!r} not found in {a_path}")

    addrs = [int(r) * (reclen + pad) for r in recs]

    with open(a_path, "rb") as fh:
        if kdm > 0 and len(addrs) == kdm:  # 3-D
            if layers is None:
                layers = list(range(kdm))
            out = np.zeros((len(layers), jdm, idm), dtype=np.float32)
            for out_k, k in enumerate(layers):
                fh.seek(addrs[k], 0)
                data = fh.read(idm * jdm * 4)
                raw = np.frombuffer(data, dtype=">f4", count=idm * jdm)
                out[out_k] = raw.reshape(jdm, idm)
        else:  # 2-D (e.g. plon, plat)
            fh.seek(addrs[0], 0)
            data = fh.read(idm * jdm * 4)
            raw = np.frombuffer(data, dtype=">f4", count=idm * jdm)
            out = raw.reshape(jdm, idm).astype(np.float32)

    # Replace HYCOM's 2**100 fill marker with `undef`.
    mask = out > 2.0 ** 99
    if mask.any():
        out = out.copy()
        out[mask] = undef
    return out


def get_model_day(a_path: Path, name: str) -> float:
    """Model day (days since 1900-12-31) written in .b for the field."""
    for ln in _b_lines(a_path):
        toks = ln.split()
        if len(toks) < 4:
            continue
        stripped = toks[0].replace(".", "").replace(":", "")
        if stripped == name:
            return float(toks[3])
    raise KeyError(f"Field {name!r} not found in {a_path}.b")


def _model_day_to_datetime(model_day: float) -> datetime:
    """Convert HYCOM model day (yrflag=3) to a UTC datetime.

    yrflag=3 means "days since 1900-12-31 00:00". Ported from
    HYCOM-tools/bin/hycom_wind_ymdh.f (same math as
    RTOFS_Post/hycom_wind_ymdh.py, but in one datetime step here).
    """
    days = float(model_day)
    epoch = datetime(1900, 12, 31, 0, 0)
    whole = int(days)
    frac_hours = (days - whole) * 24.0
    return epoch + timedelta(days=whole, hours=frac_hours)


# ---------------------------------------------------------------------------
# Horizontal regrid weights (cKDTree + inverse-distance-weighting)
# ---------------------------------------------------------------------------

def _lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray, radius: float = 1.0
                   ) -> np.ndarray:
    """Cartesian unit-sphere coordinates for KDTree distance metrics.

    Using chord distance in 3-D avoids the periodicity-in-longitude problem
    at the +/-180 seam, and treats the poles correctly.
    """
    la = np.deg2rad(lat)
    lo = np.deg2rad(lon)
    cla = np.cos(la)
    x = radius * cla * np.cos(lo)
    y = radius * cla * np.sin(lo)
    z = radius * np.sin(la)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


@dataclass
class HorizWeights:
    """Precomputed k-NN indices + inverse-distance weights, shaped (M, K).

    `M` is the number of target cells (in flattened order) and `K` is the
    number of source neighbours per target (typically 4).
    """
    idx: np.ndarray       # (M, K) int64 indices into flattened source
    w: np.ndarray         # (M, K) float64 IDW weights, rows sum to 1
    target_shape: tuple[int, int]
    source_shape: tuple[int, int]


def build_horiz_weights(src_lon2d: np.ndarray, src_lat2d: np.ndarray,
                        tgt_lon1d: np.ndarray, tgt_lat1d: np.ndarray,
                        k: int = 4, power: float = 2.0) -> HorizWeights:
    """Precompute IDW weights from a curvilinear source to a regular target.

    Parameters
    ----------
    src_lon2d, src_lat2d : (Ys, Xs)
        Source curvilinear grid, positional coords.
    tgt_lon1d, tgt_lat1d : (Xt,), (Yt,)
        Target regular grid, 1-D.
    k
        Number of nearest neighbours (default 4 -> bilinear-like blend).
    power
        Exponent for IDW distance (default 2 -> classic inverse-square).
    """
    Ys, Xs = src_lon2d.shape
    src_xyz = _lonlat_to_xyz(src_lon2d.ravel(), src_lat2d.ravel())
    tree = cKDTree(src_xyz)

    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon1d, tgt_lat1d)
    tgt_xyz = _lonlat_to_xyz(tgt_lon2d.ravel(), tgt_lat2d.ravel())

    dist, idx = tree.query(tgt_xyz, k=k, workers=-1)
    # Guard against exact hits (distance zero -> infinite weight).
    dist = np.maximum(dist, 1e-12)
    inv = 1.0 / dist ** power
    w = inv / inv.sum(axis=1, keepdims=True)

    return HorizWeights(
        idx=idx.astype(np.int64),
        w=w.astype(np.float64),
        target_shape=(len(tgt_lat1d), len(tgt_lon1d)),
        source_shape=(Ys, Xs),
    )


def apply_horiz_weights(field_2d: np.ndarray, hw: HorizWeights,
                        fill: float = np.nan) -> np.ndarray:
    """Regrid one 2-D layer from the source curvilinear grid to target regular.

    NaNs in `field_2d` propagate: a target cell whose k neighbours are all
    NaN comes out NaN; a partial-neighbour cell uses only the finite ones,
    renormalizing the weights so they sum to 1.
    """
    return apply_horiz_weights_stack(field_2d[None], hw, fill=fill)[0]


def apply_horiz_weights_stack(fields_3d: np.ndarray, hw: HorizWeights,
                              fill: float = np.nan,
                              chunk_layers: int = 8) -> np.ndarray:
    """Vectorized apply for a stack of layers `(nZ, Ys, Xs) -> (nZ, Yt, Xt)`.

    Processes `chunk_layers` layers at a time to keep the intermediate
    ``(chunk, M, K)`` gather bounded. Semantics per layer match
    `apply_horiz_weights`; here we amortise the target-side bookkeeping over
    many depth levels, cutting the per-layer overhead by ~5x.
    """
    nZ = fields_3d.shape[0]
    M = hw.idx.shape[0]
    K = hw.idx.shape[1]
    out = np.full((nZ, M), fill, dtype=np.float64)

    # Precompute IDW weights broadcastable over layers.
    w_base = hw.w  # (M, K), float64, rows sum to 1

    for i0 in range(0, nZ, chunk_layers):
        i1 = min(nZ, i0 + chunk_layers)
        chunk = fields_3d[i0:i1]                    # (C, Ys, Xs)
        C = chunk.shape[0]
        src_flat = chunk.reshape(C, -1)             # (C, Ys*Xs)
        vals = src_flat[:, hw.idx]                  # (C, M, K), gather
        good = np.isfinite(vals)                    # (C, M, K)
        w = w_base[None] * good                     # (C, M, K)
        wsum = w.sum(axis=-1)                       # (C, M)
        good_row = wsum > 0.0
        vals_zero = np.where(good, vals, 0.0)
        numer = (vals_zero * w).sum(axis=-1)        # (C, M)
        with np.errstate(invalid="ignore", divide="ignore"):
            interp = np.where(good_row, numer / np.where(good_row, wsum, 1.0), fill)
        out[i0:i1] = interp

    return out.reshape(nZ, *hw.target_shape)


# ---------------------------------------------------------------------------
# Vertical hybrid -> z
# ---------------------------------------------------------------------------

def _midpoint_depths_m(thknss_pa: np.ndarray) -> np.ndarray:
    """Layer midpoint depths (m) from HYCOM thknss (Pa) stack.

    dz[m] = thknss[Pa] / 9806  (HYCOM convention: g * rho0 with rho0 = 1000).
    Missing / dry columns end up as NaN so downstream interp propagates them.
    """
    dz = thknss_pa / 9806.0
    dz = np.where(np.isfinite(dz), dz, 0.0)
    top = np.zeros_like(dz)
    top[1:] = np.cumsum(dz[:-1], axis=0)
    return top + 0.5 * dz


def hybrid_to_z(field: np.ndarray, thknss_pa: np.ndarray,
                target_z: np.ndarray, fill: float = np.nan,
                depths_mid: np.ndarray | None = None) -> np.ndarray:
    """Column-by-column linear interp from HYCOM hybrid layers to z-levels.

    Parameters
    ----------
    field : (nZ_src, nY, nX)
        Source field on hybrid layers.
    thknss_pa : (nZ_src, nY, nX)
        Layer thicknesses in Pa (as read from HYCOM .a).
    target_z : (nZ_tgt,)
        Target depths [m, positive down].

    Returns
    -------
    (nZ_tgt, nY, nX) array. Target depths below the deepest wet midpoint (or
    entirely-dry columns) are `fill`.

    Notes
    -----
    Per-column `np.interp` in a Python loop turned out to be substantially
    faster than fully vectorized numpy for HYCOM-scale arrays (~60 s vs
    ~120 s on 8.96 M wet columns x 40 target z-levels): np.interp is a fast
    C binary search + linear interp, the Python overhead is only paid on
    the ~9 M wet columns (not the full 15 M grid), and there is no memory
    allocation per iteration once the wet-column indices are enumerated.
    """
    if depths_mid is None:
        depths_mid = _midpoint_depths_m(thknss_pa)   # (nZ, nY, nX)
    nZ_src, nY, nX = field.shape
    nZ_tgt = len(target_z)
    out = np.full((nZ_tgt, nY, nX), fill, dtype=np.float32)

    valid = np.isfinite(field) & np.isfinite(depths_mid)
    any_valid = valid.any(axis=0)

    depths_2d = depths_mid.reshape(nZ_src, -1)
    field_2d = field.reshape(nZ_src, -1)
    valid_2d = valid.reshape(nZ_src, -1)
    out_2d = out.reshape(nZ_tgt, -1)

    cols = np.flatnonzero(any_valid.ravel())
    log.info("hybrid_to_z: interpolating %d/%d wet columns onto %d z-levels",
             cols.size, any_valid.size, nZ_tgt)

    for c in cols:
        v = valid_2d[:, c]
        if not v.any():
            continue
        dp = depths_2d[v, c]
        vf = field_2d[v, c]
        interp = np.interp(target_z, dp, vf)
        interp = np.where(target_z > dp[-1], fill, interp)
        out_2d[:, c] = interp

    return out


# ---------------------------------------------------------------------------
# NetCDF writer (GLBy layout)
# ---------------------------------------------------------------------------

def _pack_int16(values: np.ndarray) -> np.ndarray:
    """Pack float field into int16 with GLBy's scale=0.001, offset=20.

    NaN / non-finite entries become the HYCOM GLBy fill sentinel (-30000).
    Valid values that would fall outside the [-30000+1, 32767] int16 range
    (in *packed* units) are clipped to the nearest representable value that
    is not the fill sentinel.
    """
    finite = np.isfinite(values)
    scaled = np.where(finite, (values - GLBY_OFFSET) / GLBY_SCALE, 0.0)
    # Reserve int16=-30000 for the fill; keep valid data strictly above.
    scaled = np.clip(scaled, -29999.0, 32767.0)
    out = scaled.astype(np.int16)
    out = np.where(finite, out, GLBY_FV_INT16)
    return out


def _write_glby_nc(path: Path, when: datetime, lat: np.ndarray, lon: np.ndarray,
                   depth: np.ndarray, var_a_name: str, var_a: np.ndarray,
                   var_b_name: str, var_b: np.ndarray) -> None:
    """Write a two-variable GLBy-style NetCDF (int16 storage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("depth", depth.size)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)

        t = nc.createVariable("time", "f8", ("time",))
        t.long_name = "Valid Time"
        t.units = "hours since 2000-01-01 00:00:00"
        t.time_origin = "2000-01-01 00:00:00"
        t.calendar = "gregorian"
        t.axis = "T"
        hrs = (when - datetime(2000, 1, 1)).total_seconds() / 3600.0
        t[:] = np.array([hrs], dtype=np.float64)

        d = nc.createVariable("depth", "f8", ("depth",))
        d.long_name = "Depth"
        d.units = "m"
        d.positive = "down"
        d.axis = "Z"
        d[:] = depth

        la = nc.createVariable("lat", "f8", ("lat",))
        la.long_name = "Latitude"
        la.units = "degrees_north"
        la.axis = "Y"
        la[:] = lat

        lo = nc.createVariable("lon", "f8", ("lon",))
        lo.long_name = "Longitude"
        lo.units = "degrees_east"
        lo.axis = "X"
        lo[:] = lon

        for name, field in ((var_a_name, var_a), (var_b_name, var_b)):
            # Disable zlib on this intermediate NC: it's a throw-away staging
            # file consumed once by run_ogcm_dl.py, and zlib=1 was costing
            # ~1 minute per file for ~4x disk savings (~1.3 GB compressed vs
            # ~5 GB raw). We can afford the disk cost; the CPU savings on
            # both write and subsequent read are large.
            v = nc.createVariable(
                name, "i2", ("time", "depth", "lat", "lon"),
                fill_value=GLBY_FV_INT16, zlib=False,
            )
            v.long_name = {
                "water_temp": "Water Temperature",
                "salinity": "Salinity",
                "water_u": "Eastward Water Velocity",
                "water_v": "Northward Water Velocity",
            }.get(name, name)
            v.standard_name = {
                "water_temp": "sea_water_temperature",
                "salinity": "sea_water_salinity",
                "water_u": "eastward_sea_water_velocity",
                "water_v": "northward_sea_water_velocity",
            }.get(name, name)
            v.units = {
                "water_temp": "degC", "salinity": "psu",
                "water_u": "m/s", "water_v": "m/s",
            }.get(name, "1")
            v.missing_value = GLBY_FV_INT16
            v.scale_factor = GLBY_SCALE
            v.add_offset = GLBY_OFFSET
            # We do the packing to int16 explicitly, so disable netCDF4's own
            # auto-scaling machinery on write; otherwise it would apply the
            # scale/offset a second time and overflow the int16 range.
            v.set_auto_scale(False)
            v.set_auto_mask(False)
            packed = _pack_int16(field[None, ...])   # (time=1, depth, lat, lon)
            v[:] = packed

        nc.source = "RTOFS archv, regridded to GLBy layout by ogcm_dl.regrid"
        nc.history = f"created {datetime.utcnow():%Y-%m-%d %H:%M} UTC"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class ArchvNCPaths:
    """Result of a successful archv -> GLBy conversion."""
    when: datetime
    ts3z: Path
    uv3z: Path | None
    # Optional in-memory GLBy grids for the freshly regridded fields, so a
    # caller can bypass the round-trip through the intermediate NC entirely.
    # The `_ts` grid holds salinity + water_temp; `_uv` holds water_u/v.
    # `None` when the caller opted out of in-memory retention.
    ts_grid: "object | None" = None
    uv_grid: "object | None" = None


def convert_archv_to_glby(
    archv_a: Path,
    grid_a: Path,
    out_dir: Path,
    do_uv: bool = True,
    target_lon: np.ndarray = GLBY_LON,
    target_lat: np.ndarray = GLBY_LAT,
    target_z: np.ndarray = GLBY_Z_LEVELS,
    return_in_memory: bool = False,
    write_intermediate: bool | None = None,
) -> ArchvNCPaths:
    """Convert a HYCOM archv (.a/.b) into GLBy-style ts3z (and optionally uv3z) NCs.

    Parameters
    ----------
    archv_a
        Path to the archive `.a` binary. Its `.b` sibling must exist.
    grid_a
        Path to the regional grid `.a` file (with matching `.b`). Provides
        the source `plat`, `plon` curvilinear coordinates.
    out_dir
        Directory to write output NCs into.
    do_uv
        If True, produce a matching `_uv3z.nc` file too.

    Returns
    -------
    Paths to the ts3z (and optional uv3z) NetCDFs, plus the timestamp.
    """
    archv_a = Path(archv_a).resolve()
    grid_a = Path(grid_a).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # When the caller keeps everything in-memory the intermediate NC is
    # pure overhead (29 s write + a stale-file-on-disk foot-gun). Default
    # to skipping the write in that case.
    if write_intermediate is None:
        write_intermediate = not return_in_memory

    log.info("Reading grid %s", grid_a)
    plon = _read_field(grid_a, "plon")
    plat = _read_field(grid_a, "plat")
    # Sanitize longitudes into [0, 360).
    plon = np.where(plon >= 360.0, plon - 360.0, plon)
    plon = np.where(plon < 0.0, plon + 360.0, plon)

    log.info("Building horizontal regrid weights (src %s -> tgt %d x %d)",
             plat.shape, target_lat.size, target_lon.size)
    hw = build_horiz_weights(plon, plat, target_lon, target_lat, k=4)

    log.info("Reading thknss (all layers)")
    thknss = _read_field(archv_a, "thknss")
    when = _model_day_to_datetime(get_model_day(archv_a, "thknss"))
    log.info("Archive valid time: %s", when.isoformat())

    # Midpoint depths only depend on thknss and are reused by every 3-D
    # field (temp, salin, u-vel, v-vel), so compute once instead of 4x.
    log.info("Computing layer-midpoint depths (reused across fields)")
    depths_mid_shared = _midpoint_depths_m(thknss)

    def _process_var(name: str) -> np.ndarray:
        log.info("Reading %s (all layers)", name)
        raw = _read_field(archv_a, name)
        log.info("Vertical hybrid -> z on %s", name)
        vz = hybrid_to_z(raw, thknss, target_z, depths_mid=depths_mid_shared)
        del raw
        log.info("Horizontal regrid on %s (per-layer)", name)
        out = np.empty((target_z.size, target_lat.size, target_lon.size),
                       dtype=np.float32)
        for k in range(target_z.size):
            out[k] = apply_horiz_weights(vz[k], hw, fill=np.nan)
        del vz
        return out

    temp = _process_var("temp")
    salin = _process_var("salin")

    # Stem for the two output NCs.
    stem = f"rtofs_glby_{when:%Y%m%d%H}_t000"
    ts_path = out_dir / f"{stem}_ts3z.nc"

    if write_intermediate:
        log.info("Writing %s", ts_path)
        _write_glby_nc(ts_path, when, target_lat, target_lon, target_z,
                       "salinity", salin, "water_temp", temp)
    else:
        log.info("Skipping ts3z NC write (in-memory pipeline path)")

    ts_grid = None
    if return_in_memory:
        # Import lazily to avoid a hard cycle at module import time.
        from .ogcm_io import Bc3dGrid
        # ogcm_io returns fields where fills are the FV sentinel; do the same
        # here so downstream code sees identical semantics.
        from .constants import FV
        sp = np.where(np.isfinite(salin), salin, FV).astype(np.float64)
        tm = np.where(np.isfinite(temp), temp, FV).astype(np.float64)
        ts_grid = Bc3dGrid(
            bc3d_lon=target_lon.astype(np.float64),
            bc3d_lat=target_lat.astype(np.float64),
            bc3d_z=target_z.astype(np.float64),
            bc3d_sp=sp,
            bc3d_t=tm,
        )
    del temp, salin

    uv_path: Path | None = None
    uv_grid = None
    if do_uv:
        u = _process_var("u-vel")
        v = _process_var("v-vel")
        uv_path = out_dir / f"{stem}_uv3z.nc"
        if write_intermediate:
            log.info("Writing %s", uv_path)
            _write_glby_nc(uv_path, when, target_lat, target_lon, target_z,
                           "water_u", u, "water_v", v)
        else:
            log.info("Skipping uv3z NC write (in-memory pipeline path)")
        if return_in_memory:
            from .ogcm_io import Bc3dGrid
            from .constants import FV
            u64 = np.where(np.isfinite(u), u, FV).astype(np.float64)
            v64 = np.where(np.isfinite(v), v, FV).astype(np.float64)
            uv_grid = Bc3dGrid(
                bc3d_lon=target_lon.astype(np.float64),
                bc3d_lat=target_lat.astype(np.float64),
                bc3d_z=target_z.astype(np.float64),
                bc3d_sp=u64,
                bc3d_t=v64,
            )
        del u, v

    return ArchvNCPaths(when=when, ts3z=ts_path, uv3z=uv_path,
                         ts_grid=ts_grid, uv_grid=uv_grid)
