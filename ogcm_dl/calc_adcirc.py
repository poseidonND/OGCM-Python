"""ADCIRC-grid science calculations.

Ports `Calc_BC2D_TS_ADCIRC`, `Calc_BC2D_UV_ADCIRC`, and `Calc_Steric_Adjustment`
from OGCM_DL.f90. Indices are 0-based throughout.

The TS routine returns the depth-averaged baroclinic pressure gradients (BPGX,
BPGY), surface sigma-t (SigTS), seabed buoyancy frequency (NB), depth-averaged
buoyancy frequency (NM), and mixed-layer depth ratio (MLD).

The UV routine returns the depth-averaged momentum dispersion (DispX, DispY).

The Steric routine returns the baroclinic sea level (BCSL).

All three preserve the per-corner GSW evaluation order of the Fortran (compute
SA at each of the 4 corners separately before bilinear interpolation, so that
SA, CT and the resulting density are identical to the Fortran results).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import gsw  # type: ignore[import-untyped]
import numpy as np

log = logging.getLogger(__name__)

from ._mlp import mlp as _gsw_mlp
from .constants import DFV, FV, FVP, G, RHO_WAT0
from .interp import InterpWeights
from .mesh import AdcircMesh
from .ogcm_io import Bc3dGrid

# Default node-chunk size for the ADCIRC-side calculators. Peak memory in
# the four-corner gather + GSW block scales like ~O(chunk * NZ * 300 bytes),
# so 100_000 keeps peak under ~1.2 GB at NZ=40 regardless of total mesh
# size. Override via the `np_chunk` kwarg on any calc_*_adcirc call, or
# via the pipeline wrappers' `--np-chunk` flag.
NP_CHUNK_DEFAULT: int = 100_000


@dataclass
class TsAdcResult:
    """Output of `calc_bc2d_ts_adcirc`."""
    bpg_adc_x: np.ndarray   # (NP,)
    bpg_adc_y: np.ndarray   # (NP,)
    sigts_adc: np.ndarray   # (NP,)
    nb_adc: np.ndarray      # (NP,)
    nm_adc: np.ndarray      # (NP,)
    mld_adc: np.ndarray     # (NP,)


@dataclass
class UvAdcResult:
    """Output of `calc_bc2d_uv_adcirc`."""
    dispx_adc: np.ndarray   # (NP,)
    dispy_adc: np.ndarray   # (NP,)


@dataclass
class StericAdcResult:
    """Output of `calc_steric_adjustment`."""
    bcsl_adc: np.ndarray    # (NP,)


# ---------------------------------------------------------------------------
# Helpers: gather the 4 corner values for every node, vectorized over (NZ, NP)
# ---------------------------------------------------------------------------

def _gather_corners(field: np.ndarray, weights: InterpWeights,
                    sl: slice | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 4 corner stacks of `field` shaped (NZ, NP).

    `field` is shape (NZ, NY, NX). `weights.indxy` packs (i, j, ir, jr) per
    node, where (i, j) is the lower-left corner and (ir, jr) is the upper-right.

    If `sl` is given, only the nodes in that slice are gathered -- returned
    arrays have shape (NZ, len(sl)). This is what lets `calc_bc2d_*_adcirc`
    chunk over nodes and keep peak memory bounded.
    """
    if sl is None:
        sl = slice(None)
    i_left = weights.indxy[0, sl]
    j_low = weights.indxy[1, sl]
    i_right = weights.indxy[2, sl]
    j_high = weights.indxy[3, sl]
    f1 = field[:, j_low, i_left]    # (NZ, chunk), corner 1: (i_left, j_low)
    f2 = field[:, j_low, i_right]   # corner 2: (i_right, j_low)
    f3 = field[:, j_high, i_left]   # corner 3: (i_left, j_high)
    f4 = field[:, j_high, i_right]  # corner 4: (i_right, j_high)
    return f1, f2, f3, f4


def _interp_to_node(stacks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    weights: np.ndarray) -> np.ndarray:
    """Apply bilinear weights to (NZ, NP) corner stacks; result shape (NZ, NP)."""
    w1 = weights[0][None, :]
    w2 = weights[1][None, :]
    w3 = weights[2][None, :]
    w4 = weights[3][None, :]
    return stacks[0] * w1 + stacks[1] * w2 + stacks[2] * w3 + stacks[3] * w4


def _gsw_sa_at_corners(sp_corners, z, lon_corners, lat_corners
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run gsw.SA_from_SP at every corner -- vectorized over (NZ, NP)."""
    z_2d = z[:, None]
    sp_clipped = [np.maximum(2.0, s) for s in sp_corners]
    sa = []
    for sp_c, lon_c, lat_c in zip(sp_clipped, lon_corners, lat_corners):
        sa.append(gsw.SA_from_SP(sp_c, z_2d, lon_c[None, :], lat_c[None, :]))
    return tuple(sa)  # type: ignore[return-value]


def _gsw_ct_at_corners(sa_corners, t_corners, z
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run gsw.CT_from_t at every corner -- vectorized over (NZ, NP)."""
    z_2d = z[:, None]
    return tuple(
        gsw.CT_from_t(sa, t, z_2d) for sa, t in zip(sa_corners, t_corners)
    )  # type: ignore[return-value]


def _build_corner_lonlat(weights: InterpWeights,
                         bc3d_lon: np.ndarray,
                         bc3d_lat: np.ndarray,
                         sl: slice | None = None
                         ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return corner lon and lat arrays, each tuple shape (4, chunk).

    If `sl` is given, only the nodes in that slice are extracted.
    """
    if sl is None:
        sl = slice(None)
    i_left = weights.indxy[0, sl]
    j_low = weights.indxy[1, sl]
    i_right = weights.indxy[2, sl]
    j_high = weights.indxy[3, sl]
    lon = (
        bc3d_lon[i_left],
        bc3d_lon[i_right],
        bc3d_lon[i_left],
        bc3d_lon[i_right],
    )
    lat = (
        bc3d_lat[j_low],
        bc3d_lat[j_low],
        bc3d_lat[j_high],
        bc3d_lat[j_high],
    )
    return lon, lat


def _build_valid_mask(sp_corners, t_corners, indz: np.ndarray, nz: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Build a (NZ, NP) mask of valid layers and a (NP,) `truebottom` array.

    A layer is valid if all 8 corner values (4 SP + 4 T) are above the fill
    threshold AND the layer index is <= INDZ[ip] (i.e., layer is shallower
    than the ADCIRC node depth).

    `truebottom[ip]` is the largest valid layer index, assuming valid layers
    are contiguous from the surface down (which matches the Fortran behavior
    in practice). Returns -1 when no valid layer exists.
    """
    above_fill = np.ones(sp_corners[0].shape, dtype=bool)
    for arr in (*sp_corners, *t_corners):
        above_fill &= arr > FVP
    ks = np.arange(nz, dtype=np.int64)[:, None]
    within_depth = ks <= indz[None, :]  # INDZ is -1 for nodes too shallow
    valid = above_fill & within_depth
    truebottom = valid.sum(axis=0).astype(np.int64) - 1   # 0-based; -1 if none
    return valid, truebottom


# ---------------------------------------------------------------------------
# Calc_BC2D_TS_ADCIRC
# ---------------------------------------------------------------------------

def calc_bc2d_ts_adcirc(grid: Bc3dGrid, mesh: AdcircMesh,
                        weights: InterpWeights,
                        np_chunk: int = NP_CHUNK_DEFAULT) -> TsAdcResult:
    """Compute the TS-side ADCIRC outputs for a single time step.

    Mirrors `Calc_BC2D_TS_ADCIRC` in the Fortran program. Returns BPGX, BPGY,
    SigTS, MLD, NB, NM at every ADCIRC node.

    For memory scalability we process the NP nodes in chunks of `np_chunk`.
    Peak memory in the corner-gather + GSW block scales like
    ``chunk * NZ * ~300 bytes`` (i.e. ~1.2 GB at NZ=40, chunk=100_000), so
    the calc stays bounded no matter how big the mesh is. Set `np_chunk`
    smaller on RAM-tight machines, larger to squeeze a bit more speed.
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    z = grid.bc3d_z
    z_2d = z[:, None]
    dz_z = np.diff(z)

    # Outputs (small: (NP,) or (NZ, NP) f8 for bcp_adc; e.g. 5 M nodes -> 200 MB).
    sigts_adc = np.zeros(np_, dtype=np.float64)
    nb_adc = np.zeros(np_, dtype=np.float64)
    nm_adc = np.zeros(np_, dtype=np.float64)
    mld_adc = np.zeros(np_, dtype=np.float64)
    # bcp_adc is fully materialised across nodes because the BPG element
    # loop later on reads bcp_adc[k, nm[ie,0..2]] with arbitrary
    # cross-node access. FV-init so masked entries are recognisable.
    # float32: 2.1 GB at NP=13.4 M / NZ=40 vs 4.3 GB as float64; the
    # subsequent BPG loop promotes to Python float on read.
    bcp_adc = np.full((nz, np_), np.float32(FV), dtype=np.float32)

    for i0 in range(0, np_, np_chunk):
        i1 = min(np_, i0 + np_chunk)
        sl = slice(i0, i1)

        log.info("TS calc: nodes %d–%d / %d", i0, i1, np_)
        sp = _gather_corners(grid.bc3d_sp, weights, sl=sl)
        t = _gather_corners(grid.bc3d_t, weights, sl=sl)
        lon_c, lat_c = _build_corner_lonlat(
            weights, grid.bc3d_lon, grid.bc3d_lat, sl=sl
        )
        valid, truebottom = _build_valid_mask(
            sp, t, weights.indz[sl], nz
        )

        # GSW at 4 corners (invalid SP clipped inside `_gsw_sa_at_corners`).
        sa_c = _gsw_sa_at_corners(sp, z, lon_c, lat_c)
        ct_c = _gsw_ct_at_corners(sa_c, t, z)
        del sp, t, lon_c, lat_c   # free source-side gathers early

        # Interpolate to ADCIRC nodes (chunk-shaped)
        sa_node = _interp_to_node(sa_c, weights.weights[:, sl])
        ct_node = _interp_to_node(ct_c, weights.weights[:, sl])
        del sa_c, ct_c

        # Density from interpolated SA, CT
        rho_node = gsw.rho(sa_node, ct_node, z_2d)

        # Surface sigma-t
        sigts_adc[sl] = np.where(valid[0], rho_node[0] - RHO_WAT0, 0.0)

        # Trapezoidal cumulative (rho - RHO_WAT0) dz -> BCP_ADC chunk
        rho_safe = np.where(valid, rho_node, RHO_WAT0)
        contrib = np.zeros_like(rho_safe)
        contrib[1:] = (
            0.5 * (rho_safe[1:] + rho_safe[:-1]) - RHO_WAT0
        ) * dz_z[:, None]
        bcp_chunk = np.cumsum(contrib, axis=0)
        bcp_chunk = np.where(valid, bcp_chunk, FV)
        bcp_adc[:, sl] = bcp_chunk
        del rho_node, rho_safe, contrib, bcp_chunk

        # Per-profile NB, NM, MLD.
        wet = truebottom >= 1
        if wet.any():
            sa_m = np.where(valid, sa_node, np.nan)
            ct_m = np.where(valid, ct_node, np.nan)
            lat_row = mesh.sfea[sl]
            try:
                n2, _ = gsw.Nsquared(sa_m, ct_m, z[:, None], lat=lat_row)
            except Exception:
                n2 = None
            # Mixed-layer pressure: one gsw.rho for the whole chunk, then
            # the Matlab 20-dbar retry only on the shallow-MLD subset.
            rho0 = gsw.rho(sa_node, ct_node, 0.0)
            rho0 = np.where(valid, rho0, np.nan)
            min_rho = np.nanmin(rho0, axis=0)
            within = ((min_rho + 0.3) - rho0) > 0.0
            within = np.where(valid, within, False)
            k_mlp = np.where(within, np.arange(nz, dtype=np.int32)[:, None], -1).max(axis=0)
            pmin = np.nanmin(np.where(valid, z[:, None], np.nan), axis=0)
            local_wet = np.flatnonzero(wet)
            for ip_local in local_wet:
                tb = int(truebottom[ip_local])
                ip = i0 + ip_local
                if n2 is not None:
                    n2_p = np.asarray(n2[:tb, ip_local], dtype=np.float64)
                else:
                    n2_p, _ = gsw.Nsquared(
                        sa_node[: tb + 1, ip_local],
                        ct_node[: tb + 1, ip_local],
                        z[: tb + 1],
                        lat=np.full(tb + 1, mesh.sfea[ip]),
                    )
                    n2_p = np.asarray(n2_p, dtype=np.float64)
                n2_p = np.maximum(0.0, n2_p)
                nb_adc[ip] = float(np.sqrt(n2_p[-1]))
                nm_adc[ip] = float(
                    (np.diff(z[: tb + 1]) * np.sqrt(n2_p)).sum() / z[tb]
                )
                if (not np.isfinite(pmin[ip_local]) or pmin[ip_local] > 20.0
                        or k_mlp[ip_local] < 0):
                    mld_val = float("nan")
                else:
                    mlp_val = float(z[int(k_mlp[ip_local])])
                    if mlp_val - float(pmin[ip_local]) < 20.0:
                        mld_val = _gsw_mlp(
                            sa_node[: tb + 1, ip_local],
                            ct_node[: tb + 1, ip_local],
                            z[: tb + 1],
                        )
                    else:
                        mld_val = mlp_val
                if not np.isfinite(mld_val):
                    mld_val = DFV * z[tb]
                mld_adc[ip] = min(DFV, mld_val) / z[tb]
        del sa_node, ct_node, valid, truebottom

    # Element-wise BPG integration, then area-weighted to nodes
    bpg_x_node, bpg_y_node = _compute_bpg_per_node(
        mesh, weights.indz, z, bcp_adc
    )
    bpg_x_node *= G / RHO_WAT0
    bpg_y_node *= G / RHO_WAT0

    # Apply the BPG limiter and divide by total area
    has_area = mesh.total_area > 0.0
    bpg_x_node = np.where(has_area, bpg_x_node / np.where(has_area, mesh.total_area, 1.0), 0.0)
    bpg_y_node = np.where(has_area, bpg_y_node / np.where(has_area, mesh.total_area, 1.0), 0.0)

    return TsAdcResult(
        bpg_adc_x=bpg_x_node,
        bpg_adc_y=bpg_y_node,
        sigts_adc=sigts_adc,
        nb_adc=nb_adc,
        nm_adc=nm_adc,
        mld_adc=mld_adc,
    )


def _compute_bpg_per_node(mesh: AdcircMesh, indz: np.ndarray,
                          z: np.ndarray, bcp_adc: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Integrate BCP gradients over depth, then scatter to nodes.

    Same arithmetic as the Fortran `DO IE = 1,NE` loop in
    `Calc_BC2D_TS_ADCIRC`, but fully vectorized (one z-level at a time,
    chunks of elements) so a 26 M-element mesh takes seconds instead of
    tens of minutes. Returns the *un-area-normalized* BPG sums on nodes.
    """
    ne = mesh.ne
    np_ = mesh.np_
    nm = mesh.nm
    n1 = nm[:, 0]
    n2 = nm[:, 1]
    n3 = nm[:, 2]
    areas = mesh.areas
    dphi_dx = mesh.dphi_dx
    dphi_dy = mesh.dphi_dy
    dz = np.diff(z)
    nz = int(z.size)

    nmax = np.minimum(np.minimum(indz[n1], indz[n2]), indz[n3]).astype(np.int32) - 1

    bpgx_e = np.zeros(ne, dtype=np.float64)
    bpgy_e = np.zeros(ne, dtype=np.float64)
    last_bx = np.zeros(ne, dtype=np.float64)
    last_by = np.zeros(ne, dtype=np.float64)
    last_dz = np.zeros(ne, dtype=np.float64)
    I = nmax.copy()  # effective last interval; shrinks if a mid-column fill is hit

    # Walk z-levels once: add each valid interval at facval=1, then subtract
    # half of the last completed interval (Fortran's facval=0.5 on the last
    # step, and the "rollback half-step" when a fill is encountered).
    chunk = 400_000
    for i0 in range(0, ne, chunk):
        i1 = min(ne, i0 + chunk)
        sl = slice(i0, i1)
        n1s, n2s, n3s = n1[sl], n2[sl], n3[sl]
        ddx, ddy = dphi_dx[sl], dphi_dy[sl]
        I_s = I[sl]
        bx_acc = bpgx_e[sl]
        by_acc = bpgy_e[sl]
        lbx = last_bx[sl]
        lby = last_by[sl]
        ldz = last_dz[sl]

        for iz_0 in range(nz - 1):
            active = iz_0 <= I_s
            if not np.any(active):
                continue
            k = iz_0 + 1
            bcp1 = np.asarray(bcp_adc[k, n1s], dtype=np.float64)
            bcp2 = np.asarray(bcp_adc[k, n2s], dtype=np.float64)
            bcp3 = np.asarray(bcp_adc[k, n3s], dtype=np.float64)
            fill = (bcp1 < FVP) | (bcp2 < FVP) | (bcp3 < FVP)
            newfill = active & fill
            I_s = np.where(newfill, iz_0 - 1, I_s)
            add = active & ~fill
            if not np.any(add):
                continue
            bx = bcp1 * ddx[:, 0] + bcp2 * ddx[:, 1] + bcp3 * ddx[:, 2]
            by = bcp1 * ddy[:, 0] + bcp2 * ddy[:, 1] + bcp3 * ddy[:, 2]
            dzi = dz[iz_0]
            bx_acc = np.where(add, bx_acc + bx * dzi, bx_acc)
            by_acc = np.where(add, by_acc + by * dzi, by_acc)
            lbx = np.where(add, bx, lbx)
            lby = np.where(add, by, lby)
            ldz = np.where(add, dzi, ldz)

        I[sl] = I_s
        bpgx_e[sl] = bx_acc
        bpgy_e[sl] = by_acc
        last_bx[sl] = lbx
        last_by[sl] = lby
        last_dz[sl] = ldz

    good = I >= 0
    bpgx_e = np.where(good, bpgx_e - 0.5 * last_bx * last_dz, 0.0)
    bpgy_e = np.where(good, bpgy_e - 0.5 * last_by * last_dz, 0.0)
    zbot = np.zeros(ne, dtype=np.float64)
    zbot[good] = z[I[good] + 1]
    ok_z = good & (zbot > 0.0)
    bpgx_e = np.where(ok_z, bpgx_e / np.where(ok_z, zbot, 1.0), 0.0)
    bpgy_e = np.where(ok_z, bpgy_e / np.where(ok_z, zbot, 1.0), 0.0)

    contrib_x = areas * bpgx_e
    contrib_y = areas * bpgy_e
    bpg_x = np.zeros(np_, dtype=np.float64)
    bpg_y = np.zeros(np_, dtype=np.float64)
    np.add.at(bpg_x, n1, contrib_x)
    np.add.at(bpg_x, n2, contrib_x)
    np.add.at(bpg_x, n3, contrib_x)
    np.add.at(bpg_y, n1, contrib_y)
    np.add.at(bpg_y, n2, contrib_y)
    np.add.at(bpg_y, n3, contrib_y)
    return bpg_x, bpg_y


# ---------------------------------------------------------------------------
# Calc_BC2D_UV_ADCIRC
# ---------------------------------------------------------------------------

def calc_bc2d_uv_adcirc(grid: Bc3dGrid, mesh: AdcircMesh,
                        weights: InterpWeights,
                        np_chunk: int = NP_CHUNK_DEFAULT) -> UvAdcResult:
    """Compute the UV-side ADCIRC outputs for a single time step.

    Mirrors `Calc_BC2D_UV_ADCIRC` in the Fortran program. Returns DispX, DispY
    at every ADCIRC node.

    Note: the Fortran source has a typo where `DispX_ADC(NM3)` and
    `DispY_ADC(NM3)` are mistakenly written as `(NM2)` again. This Python port
    fixes that obvious bug by writing to all three triangle nodes -- the
    intended behavior.

    Chunks over `np_chunk` nodes at a time to keep peak memory bounded
    regardless of mesh size (see `calc_bc2d_ts_adcirc` for the rationale).
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    ne = mesh.ne
    z = grid.bc3d_z
    dz_z = np.diff(z)

    # Per-node depth-integrated dispersion pieces persist across chunks
    # so the element loop can read them by node index.
    duu_adc = np.zeros(np_, dtype=np.float64)
    dvv_adc = np.zeros(np_, dtype=np.float64)
    duv_adc = np.zeros(np_, dtype=np.float64)

    for i0 in range(0, np_, np_chunk):
        i1 = min(np_, i0 + np_chunk)
        sl = slice(i0, i1)

        u = _gather_corners(grid.bc3d_sp, weights, sl=sl)   # SP slot holds water_u
        v = _gather_corners(grid.bc3d_t, weights, sl=sl)    # T slot holds water_v
        valid, truebottom = _build_valid_mask(u, v, weights.indz[sl], nz)

        # Interpolate to ADCIRC nodes (chunk-shaped)
        uu = _interp_to_node(u, weights.weights[:, sl])
        vv = _interp_to_node(v, weights.weights[:, sl])
        del u, v
        uu = np.where(valid, uu, 0.0)
        vv = np.where(valid, vv, 0.0)

        # Trapezoidal integration -> ADC2D_U per chunk node
        u_step = 0.5 * (uu[1:] + uu[:-1]) * dz_z[:, None]
        v_step = 0.5 * (vv[1:] + vv[:-1]) * dz_z[:, None]
        u_cum = np.cumsum(u_step, axis=0)
        v_cum = np.cumsum(v_step, axis=0)
        del u_step, v_step

        chunk_size = i1 - i0
        adc2d_u = np.zeros(chunk_size, dtype=np.float64)
        adc2d_v = np.zeros(chunk_size, dtype=np.float64)
        valid_node = truebottom >= 1
        if valid_node.any():
            tb_idx = np.clip(truebottom - 1, 0, u_cum.shape[0] - 1)
            local = np.flatnonzero(valid_node)
            adc2d_u[valid_node] = (
                u_cum[tb_idx[valid_node], local] / z[truebottom[valid_node]]
            )
            adc2d_v[valid_node] = (
                v_cum[tb_idx[valid_node], local] / z[truebottom[valid_node]]
            )
        del u_cum, v_cum

        # Per-layer Udiff, Vdiff (=U(z) - U_bar)
        udiff = np.where(valid, uu - adc2d_u[None, :], 0.0)
        vdiff = np.where(valid, vv - adc2d_v[None, :], 0.0)
        del uu, vv

        # Trapezoidal Duu, Dvv, Duv on the chunk, sum over depth
        uu_avg = 0.5 * (udiff[1:] + udiff[:-1])
        vv_avg = 0.5 * (vdiff[1:] + vdiff[:-1])
        del udiff, vdiff
        duu_chunk = np.where(valid_node, (uu_avg * uu_avg * dz_z[:, None]).sum(axis=0), 0.0)
        dvv_chunk = np.where(valid_node, (vv_avg * vv_avg * dz_z[:, None]).sum(axis=0), 0.0)
        duv_chunk = np.where(valid_node, (uu_avg * vv_avg * dz_z[:, None]).sum(axis=0), 0.0)
        del uu_avg, vv_avg

        duu_adc[sl] = duu_chunk
        dvv_adc[sl] = dvv_chunk
        duv_adc[sl] = duv_chunk

    # Element loop: gradients of Duu, Dvv, Duv via element basis derivatives
    n1 = mesh.nm[:, 0]
    n2 = mesh.nm[:, 1]
    n3 = mesh.nm[:, 2]
    ddx, ddy = mesh.dphi_dx, mesh.dphi_dy
    a = mesh.areas
    duu_x = duu_adc[n1] * ddx[:, 0] + duu_adc[n2] * ddx[:, 1] + duu_adc[n3] * ddx[:, 2]
    dvv_y = dvv_adc[n1] * ddy[:, 0] + dvv_adc[n2] * ddy[:, 1] + dvv_adc[n3] * ddy[:, 2]
    duv_x = duv_adc[n1] * ddx[:, 0] + duv_adc[n2] * ddx[:, 1] + duv_adc[n3] * ddx[:, 2]
    duv_y = duv_adc[n1] * ddy[:, 0] + duv_adc[n2] * ddy[:, 1] + duv_adc[n3] * ddy[:, 2]
    dispx_contrib = (duu_x + duv_y) * a
    dispy_contrib = (duv_x + dvv_y) * a
    dispx_adc = np.zeros(np_, dtype=np.float64)
    dispy_adc = np.zeros(np_, dtype=np.float64)
    np.add.at(dispx_adc, n1, dispx_contrib)
    np.add.at(dispx_adc, n2, dispx_contrib)
    np.add.at(dispx_adc, n3, dispx_contrib)
    np.add.at(dispy_adc, n1, dispy_contrib)
    np.add.at(dispy_adc, n2, dispy_contrib)
    np.add.at(dispy_adc, n3, dispy_contrib)

    # Depth-average + area-normalize at the nodes
    has_depth = mesh.dp > 1e-3
    dispx_adc = np.where(has_depth, dispx_adc / np.where(has_depth, mesh.dp * mesh.total_area, 1.0), 0.0)
    dispy_adc = np.where(has_depth, dispy_adc / np.where(has_depth, mesh.dp * mesh.total_area, 1.0), 0.0)

    return UvAdcResult(dispx_adc=dispx_adc, dispy_adc=dispy_adc)


# ---------------------------------------------------------------------------
# Calc_Steric_Adjustment
# ---------------------------------------------------------------------------

def calc_steric_adjustment(grid: Bc3dGrid, mesh: AdcircMesh,
                           weights: InterpWeights,
                           np_chunk: int = NP_CHUNK_DEFAULT
                           ) -> StericAdcResult:
    """Compute baroclinic sea level (BCSL) on the ADCIRC mesh.

    Mirrors `Calc_Steric_Adjustment` in the Fortran. Uses
    `gsw.geo_strf_dyn_height` per profile. Node-chunked for memory
    scalability (see `calc_bc2d_ts_adcirc`).
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    z = grid.bc3d_z

    bcsl_adc = np.zeros(np_, dtype=np.float64)

    for i0 in range(0, np_, np_chunk):
        i1 = min(np_, i0 + np_chunk)
        sl = slice(i0, i1)

        sp = _gather_corners(grid.bc3d_sp, weights, sl=sl)
        t = _gather_corners(grid.bc3d_t, weights, sl=sl)
        lon_c, lat_c = _build_corner_lonlat(
            weights, grid.bc3d_lon, grid.bc3d_lat, sl=sl
        )
        _, truebottom = _build_valid_mask(sp, t, weights.indz[sl], nz)

        sa_c = _gsw_sa_at_corners(sp, z, lon_c, lat_c)
        ct_c = _gsw_ct_at_corners(sa_c, t, z)
        del sp, t, lon_c, lat_c
        sa_node = _interp_to_node(sa_c, weights.weights[:, sl])
        ct_node = _interp_to_node(ct_c, weights.weights[:, sl])
        del sa_c, ct_c

        chunk_size = i1 - i0
        for ip_local in range(chunk_size):
            tb = int(truebottom[ip_local])
            if tb < 1:
                continue
            ip = i0 + ip_local
            sa_p = sa_node[: tb + 1, ip_local]
            ct_p = ct_node[: tb + 1, ip_local]
            z_p = z[: tb + 1]
            dha = gsw.geo_strf_dyn_height(sa_p, ct_p, z_p, p_ref=z[tb])
            dz = np.diff(z_p)
            dha_avg = float(np.sum(0.5 * (dha[1:] + dha[:-1]) * dz))
            bcsl = (float(dha[0]) - dha_avg / z[tb]) / G
            if bcsl > 1.0e5:
                bcsl = 0.0
            bcsl_adc[ip] = bcsl

    return StericAdcResult(bcsl_adc=bcsl_adc)
