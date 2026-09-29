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

import gsw  # type: ignore[import-untyped]
import numpy as np

from ._mlp import mlp as _gsw_mlp
from .constants import DFV, FV, FVP, G, RHO_WAT0
from .interp import InterpWeights
from .mesh import AdcircMesh
from .ogcm_io import Bc3dGrid


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

def _gather_corners(field: np.ndarray, weights: InterpWeights
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 4 corner stacks of `field` shaped (NZ, NP).

    `field` is shape (NZ, NY, NX). `weights.indxy` packs (i, j, ir, jr) per
    node, where (i, j) is the lower-left corner and (ir, jr) is the upper-right.
    """
    i_left = weights.indxy[0]
    j_low = weights.indxy[1]
    i_right = weights.indxy[2]
    j_high = weights.indxy[3]
    f1 = field[:, j_low, i_left]    # (NZ, NP), corner 1: (i_left, j_low)
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
                         bc3d_lat: np.ndarray
                         ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return corner lon and lat arrays, each tuple shape (4, NP)."""
    i_left = weights.indxy[0]
    j_low = weights.indxy[1]
    i_right = weights.indxy[2]
    j_high = weights.indxy[3]
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
                        weights: InterpWeights) -> TsAdcResult:
    """Compute the TS-side ADCIRC outputs for a single time step.

    Mirrors `Calc_BC2D_TS_ADCIRC` in the Fortran program. Returns BPGX, BPGY,
    SigTS, MLD, NB, NM at every ADCIRC node.
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    z = grid.bc3d_z

    # 4-corner gathers shaped (NZ, NP)
    sp = _gather_corners(grid.bc3d_sp, weights)
    t = _gather_corners(grid.bc3d_t, weights)
    lon_c, lat_c = _build_corner_lonlat(weights, grid.bc3d_lon, grid.bc3d_lat)
    valid, truebottom = _build_valid_mask(sp, t, weights.indz, nz)

    # GSW conversions at the four corners (only on valid cells; we replace
    # invalid SP with the floor of 2 so the call doesn't blow up, then mask
    # results afterwards).
    sa_c = _gsw_sa_at_corners(sp, z, lon_c, lat_c)
    ct_c = _gsw_ct_at_corners(sa_c, t, z)

    # Interpolate SA, CT to the ADCIRC node
    sa_node = _interp_to_node(sa_c, weights.weights)
    ct_node = _interp_to_node(ct_c, weights.weights)

    # Density from interpolated SA, CT (matches the Fortran exactly: rho is
    # computed from interpolated SA/CT, not bilinearly interpolated itself).
    z_2d = z[:, None]
    rho_node = gsw.rho(sa_node, ct_node, z_2d)

    # Surface sigma-t (only for nodes with a valid surface layer)
    sigts_adc = np.where(valid[0], rho_node[0] - RHO_WAT0, 0.0)

    # Trapezoidal cumulative integral of (rho - RHO_WAT0) -> BCP_ADC[k, ip]
    rho_safe = np.where(valid, rho_node, RHO_WAT0)
    contrib = np.zeros_like(rho_safe)
    dz_z = np.diff(z)
    contrib[1:] = (0.5 * (rho_safe[1:] + rho_safe[:-1]) - RHO_WAT0) * dz_z[:, None]
    bcp_adc = np.cumsum(contrib, axis=0)
    bcp_adc = np.where(valid, bcp_adc, FV)

    # NB, NM, MLD per node (per-profile gsw calls -- a Python loop is fine here).
    nb_adc = np.zeros(np_, dtype=np.float64)
    nm_adc = np.zeros(np_, dtype=np.float64)
    mld_adc = np.zeros(np_, dtype=np.float64)
    for ip in range(np_):
        tb = int(truebottom[ip])  # 0-based last valid index
        if tb < 1:  # need at least 2 valid levels
            continue
        sa_p = sa_node[: tb + 1, ip]
        ct_p = ct_node[: tb + 1, ip]
        z_p = z[: tb + 1]
        lat_p = np.full(tb + 1, mesh.sfea[ip])
        n2, _ = gsw.Nsquared(sa_p, ct_p, z_p, lat=lat_p)
        nb_adc[ip] = float(np.sqrt(max(0.0, n2[-1])))
        nm_adc[ip] = float(
            (np.diff(z_p) * np.sqrt(np.maximum(0.0, n2))).sum() / z[tb]
        )
        # MLD: matches Fortran exactly (`min(DFV, mlp) / z[tb]`).
        # `gsw_mlp` is not exposed by GSW-Python on the numpy<2 line, so we
        # use a faithful pure-Python implementation in `ogcm_dl._mlp`.
        mld_val = _gsw_mlp(sa_p, ct_p, z_p)
        if not np.isfinite(mld_val):
            mld_val = DFV * z[tb]
        mld_adc[ip] = min(DFV, mld_val) / z[tb]

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
    """Element loop that integrates BCP gradients over depth.

    Reproduces the Fortran inner loop in `Calc_BC2D_TS_ADCIRC` (the section
    starting at `DO IE = 1,NE`). Returns the *un-area-normalized* BPG sums on
    nodes; the caller multiplies by g/rho0 and divides by `total_area`.
    """
    ne = mesh.ne
    np_ = mesh.np_
    nm = mesh.nm
    areas = mesh.areas
    dphi_dx = mesh.dphi_dx
    dphi_dy = mesh.dphi_dy

    bpg_x = np.zeros(np_, dtype=np.float64)
    bpg_y = np.zeros(np_, dtype=np.float64)

    for ie in range(ne):
        nm1, nm2, nm3 = int(nm[ie, 0]), int(nm[ie, 1]), int(nm[ie, 2])
        ddx1, ddx2, ddx3 = dphi_dx[ie]
        ddy1, ddy2, ddy3 = dphi_dy[ie]

        # idzmax follows the Fortran (1-based) convention:
        #   idzmax_F = min(INDZ_F[NM1..3]) - 1
        # In 0-based: idzmax_0 = min(INDZ_0[NM1..3]) - 1
        # The element loop iterates iz_F from 1 to idzmax_F (i.e., iz_0 from
        # 0 to idzmax_0). Negative idzmax => skip element.
        idzmax_0 = min(int(indz[nm1]), int(indz[nm2]), int(indz[nm3])) - 1
        if idzmax_0 < 0:
            continue

        bpgx = 0.0
        bpgy = 0.0
        bx = 0.0
        by = 0.0
        dz_last = 0.0
        for iz_0 in range(idzmax_0 + 1):
            # Fortran reads BCP_ADC at level iz+1 (deepest of the trapezoidal
            # interval [iz, iz+1]); in 0-based that's index iz_0+1.
            k = iz_0 + 1
            bcp1 = bcp_adc[k, nm1]
            bcp2 = bcp_adc[k, nm2]
            bcp3 = bcp_adc[k, nm3]
            if bcp1 < FVP or bcp2 < FVP or bcp3 < FVP:
                # Hit a fill in the middle: roll back the previous half-step.
                bpgx -= 0.5 * bx * dz_last
                bpgy -= 0.5 * by * dz_last
                idzmax_0 = iz_0 - 1
                break
            dz = z[iz_0 + 1] - z[iz_0]
            dz_last = dz
            bx = bcp1 * ddx1 + bcp2 * ddx2 + bcp3 * ddx3
            by = bcp1 * ddy1 + bcp2 * ddy2 + bcp3 * ddy3
            facval = 0.5 if iz_0 == idzmax_0 else 1.0
            bpgx += facval * bx * dz
            bpgy += facval * by * dz

        # Depth-average using z at idzmax_0+1 (matches the Fortran)
        if idzmax_0 >= 0 and z[idzmax_0 + 1] > 0:
            bpgx /= z[idzmax_0 + 1]
            bpgy /= z[idzmax_0 + 1]
        else:
            bpgx = 0.0
            bpgy = 0.0

        # NOTE: the Fortran's HYCOM-grid path clamps BPG to [-BPG_LIM,
        # BPG_LIM]; the ADCIRC-grid path does NOT, so we don't clamp here.
        a = areas[ie]
        bpg_x[nm1] += a * bpgx
        bpg_x[nm2] += a * bpgx
        bpg_x[nm3] += a * bpgx
        bpg_y[nm1] += a * bpgy
        bpg_y[nm2] += a * bpgy
        bpg_y[nm3] += a * bpgy

    return bpg_x, bpg_y


# ---------------------------------------------------------------------------
# Calc_BC2D_UV_ADCIRC
# ---------------------------------------------------------------------------

def calc_bc2d_uv_adcirc(grid: Bc3dGrid, mesh: AdcircMesh,
                        weights: InterpWeights) -> UvAdcResult:
    """Compute the UV-side ADCIRC outputs for a single time step.

    Mirrors `Calc_BC2D_UV_ADCIRC` in the Fortran program. Returns DispX, DispY
    at every ADCIRC node.

    Note: the Fortran source has a typo where `DispX_ADC(NM3)` and
    `DispY_ADC(NM3)` are mistakenly written as `(NM2)` again. This Python port
    fixes that obvious bug by writing to all three triangle nodes -- the
    intended behavior.
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    ne = mesh.ne
    z = grid.bc3d_z

    # 4-corner gathers
    u = _gather_corners(grid.bc3d_sp, weights)   # SP slot holds water_u
    v = _gather_corners(grid.bc3d_t, weights)    # T slot holds water_v
    valid, truebottom = _build_valid_mask(u, v, weights.indz, nz)

    # Interpolate to ADCIRC nodes
    uu = _interp_to_node(u, weights.weights)
    vv = _interp_to_node(v, weights.weights)
    uu = np.where(valid, uu, 0.0)
    vv = np.where(valid, vv, 0.0)

    # Trapezoidal depth integration: ADC2D_U = (1/z_tb) * integral_0^z_tb U dz
    dz_z = np.diff(z)
    u_step = 0.5 * (uu[1:] + uu[:-1]) * dz_z[:, None]
    v_step = 0.5 * (vv[1:] + vv[:-1]) * dz_z[:, None]
    u_cum = np.cumsum(u_step, axis=0)  # (NZ-1, NP); index k is z[k+1]
    v_cum = np.cumsum(v_step, axis=0)

    adc2d_u = np.zeros(np_, dtype=np.float64)
    adc2d_v = np.zeros(np_, dtype=np.float64)
    valid_node = truebottom >= 1
    if valid_node.any():
        # Use truebottom-1 to index into u_cum/v_cum
        tb_idx = np.clip(truebottom - 1, 0, u_cum.shape[0] - 1)
        adc2d_u[valid_node] = u_cum[tb_idx[valid_node], np.flatnonzero(valid_node)] / z[truebottom[valid_node]]
        adc2d_v[valid_node] = v_cum[tb_idx[valid_node], np.flatnonzero(valid_node)] / z[truebottom[valid_node]]

    # Per-layer Udiff, Vdiff (=U(z) - U_bar)
    udiff = uu - adc2d_u[None, :]
    vdiff = vv - adc2d_v[None, :]
    udiff = np.where(valid, udiff, 0.0)
    vdiff = np.where(valid, vdiff, 0.0)

    # Trapezoidal Duu, Dvv, Duv (per-node depth-integrated dispersion)
    uu_avg = 0.5 * (udiff[1:] + udiff[:-1])
    vv_avg = 0.5 * (vdiff[1:] + vdiff[:-1])
    duu_step = uu_avg * uu_avg * dz_z[:, None]
    dvv_step = vv_avg * vv_avg * dz_z[:, None]
    duv_step = uu_avg * vv_avg * dz_z[:, None]
    duu_adc = np.where(valid_node, duu_step.sum(axis=0), 0.0)
    dvv_adc = np.where(valid_node, dvv_step.sum(axis=0), 0.0)
    duv_adc = np.where(valid_node, duv_step.sum(axis=0), 0.0)

    # Element loop: gradients of Duu, Dvv, Duv via element basis derivatives
    dispx_adc = np.zeros(np_, dtype=np.float64)
    dispy_adc = np.zeros(np_, dtype=np.float64)

    nm = mesh.nm
    areas = mesh.areas
    dphi_dx = mesh.dphi_dx
    dphi_dy = mesh.dphi_dy

    for ie in range(ne):
        nm1, nm2, nm3 = int(nm[ie, 0]), int(nm[ie, 1]), int(nm[ie, 2])
        ddx1, ddx2, ddx3 = dphi_dx[ie]
        ddy1, ddy2, ddy3 = dphi_dy[ie]
        a = areas[ie]

        duu_x = duu_adc[nm1] * ddx1 + duu_adc[nm2] * ddx2 + duu_adc[nm3] * ddx3
        dvv_y = dvv_adc[nm1] * ddy1 + dvv_adc[nm2] * ddy2 + dvv_adc[nm3] * ddy3
        duv_x = duv_adc[nm1] * ddx1 + duv_adc[nm2] * ddx2 + duv_adc[nm3] * ddx3
        duv_y = duv_adc[nm1] * ddy1 + duv_adc[nm2] * ddy2 + duv_adc[nm3] * ddy3

        dispx_contrib = (duu_x + duv_y) * a
        dispy_contrib = (duv_x + dvv_y) * a
        # Spread to all three triangle nodes (Fortran source has a typo:
        # DispX_ADC(NM2) is repeated in place of DispX_ADC(NM3)). We fix it.
        dispx_adc[nm1] += dispx_contrib
        dispx_adc[nm2] += dispx_contrib
        dispx_adc[nm3] += dispx_contrib
        dispy_adc[nm1] += dispy_contrib
        dispy_adc[nm2] += dispy_contrib
        dispy_adc[nm3] += dispy_contrib

    # Depth-average + area-normalize at the nodes
    has_depth = mesh.dp > 1e-3
    dispx_adc = np.where(has_depth, dispx_adc / np.where(has_depth, mesh.dp * mesh.total_area, 1.0), 0.0)
    dispy_adc = np.where(has_depth, dispy_adc / np.where(has_depth, mesh.dp * mesh.total_area, 1.0), 0.0)

    return UvAdcResult(dispx_adc=dispx_adc, dispy_adc=dispy_adc)


# ---------------------------------------------------------------------------
# Calc_Steric_Adjustment
# ---------------------------------------------------------------------------

def calc_steric_adjustment(grid: Bc3dGrid, mesh: AdcircMesh,
                           weights: InterpWeights) -> StericAdcResult:
    """Compute baroclinic sea level (BCSL) on the ADCIRC mesh.

    Mirrors `Calc_Steric_Adjustment` in the Fortran. Uses
    `gsw.geo_strf_dyn_height` per profile.
    """
    nz = grid.bc3d_z.size
    np_ = mesh.np_
    z = grid.bc3d_z

    sp = _gather_corners(grid.bc3d_sp, weights)
    t = _gather_corners(grid.bc3d_t, weights)
    lon_c, lat_c = _build_corner_lonlat(weights, grid.bc3d_lon, grid.bc3d_lat)
    valid, truebottom = _build_valid_mask(sp, t, weights.indz, nz)

    sa_c = _gsw_sa_at_corners(sp, z, lon_c, lat_c)
    ct_c = _gsw_ct_at_corners(sa_c, t, z)
    sa_node = _interp_to_node(sa_c, weights.weights)
    ct_node = _interp_to_node(ct_c, weights.weights)

    bcsl_adc = np.zeros(np_, dtype=np.float64)
    for ip in range(np_):
        tb = int(truebottom[ip])
        if tb < 1:
            continue
        sa_p = sa_node[: tb + 1, ip]
        ct_p = ct_node[: tb + 1, ip]
        z_p = z[: tb + 1]
        # Dynamic height anomaly w.r.t. the bottom
        dha = gsw.geo_strf_dyn_height(sa_p, ct_p, z_p, p_ref=z[tb])
        # Trapezoidal mean of DHA(z) over [0, z[tb]]
        dz = np.diff(z_p)
        dha_avg = float(np.sum(0.5 * (dha[1:] + dha[:-1]) * dz))
        bcsl = (float(dha[0]) - dha_avg / z[tb]) / G
        if bcsl > 1.0e5:
            bcsl = 0.0
        bcsl_adc[ip] = bcsl

    return StericAdcResult(bcsl_adc=bcsl_adc)
