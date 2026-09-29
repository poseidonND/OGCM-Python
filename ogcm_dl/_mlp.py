"""Pure-Python implementation of GSW `mlp` (mixed-layer pressure).

GSW-Python (3.6.20 line, the latest one compatible with `numpy<2`) does not
expose `gsw_mlp` because it is a profile-level routine and was never wrapped
through the C ufunc generator. We re-implement it here, faithful to the
canonical `gsw_mlp.m` from the GSW-Matlab toolbox (v3.06.12).

Algorithm (de Boyer Montegut et al. 2004 + GSW 0.3 kg/m^3 density-difference
threshold):

  1. Compute potential density at the surface, rho0 = rho(SA, CT, 0), for the
     full profile.
  2. Drop NaN levels, sort by pressure, deduplicate.
  3. If the shallowest pressure is deeper than 20 dbar, return NaN.
  4. mlp = the deepest pressure where (min(rho0) + 0.3) - rho0 > 0
     (i.e. still within 0.3 kg/m^3 of the surface density).
  5. If mlp - min(p) < 20 dbar, replace top-of-cast (above the bottle nearest
     5 dbar) with that bottle's SA, CT, recompute rho0 and mlp. If still
     < 20 dbar, return NaN.

The function returns a discrete bottle pressure (no linear interpolation),
matching the Matlab reference exactly.
"""

from __future__ import annotations

import gsw  # type: ignore[import-untyped]
import numpy as np


def mlp(SA: np.ndarray, CT: np.ndarray, p: np.ndarray) -> float:
    """Compute mixed-layer pressure for a single profile.

    Parameters
    ----------
    SA : array_like, shape (NZ,)
        Absolute Salinity [g/kg].
    CT : array_like, shape (NZ,)
        Conservative Temperature (ITS-90) [deg C].
    p  : array_like, shape (NZ,)
        Sea pressure [dbar] (== absolute pressure - 10.1325).

    Returns
    -------
    float
        Mixed-layer pressure [dbar], or NaN if undefined.
    """
    SA_ = np.asarray(SA, dtype=np.float64).ravel()
    CT_ = np.asarray(CT, dtype=np.float64).ravel()
    p_ = np.asarray(p, dtype=np.float64).ravel()
    if SA_.size == 0:
        return float("nan")

    rho0 = gsw.rho(SA_, CT_, np.zeros_like(SA_))

    inn = np.isfinite(rho0) & np.isfinite(SA_) & np.isfinite(CT_) & np.isfinite(p_)
    if not inn.any():
        return float("nan")

    p_in = p_[inn]
    rho0_in = rho0[inn]
    SA_in = SA_[inn]
    CT_in = CT_[inn]

    # Sort by pressure, then dedupe (Matlab: sort + unique).
    order = np.argsort(p_in, kind="stable")
    p_s = p_in[order]
    rho0_s = rho0_in[order]
    SA_s = SA_in[order]
    CT_s = CT_in[order]
    p_tmp, uniq_idx = np.unique(p_s, return_index=True)
    rho0_tmp = rho0_s[uniq_idx]
    SA_tmp = SA_s[uniq_idx]
    CT_tmp = CT_s[uniq_idx]

    if p_tmp.min() > 20.0:
        return float("nan")

    min_rho0 = float(np.min(rho0_tmp))
    diff = (min_rho0 + 0.3) - rho0_tmp
    hits = np.flatnonzero(diff > 0.0)
    if hits.size == 0:
        return float("nan")
    mlp_val = float(p_tmp[hits[-1]])

    dmlp = mlp_val - float(p_tmp.min())

    if dmlp < 20.0:
        # Replace bottles above the bottle nearest 5 dbar with that bottle.
        i4 = int(np.argmin(np.abs(p_tmp - 5.0)))
        if i4 >= 1:
            SA_tmp = SA_tmp.copy()
            CT_tmp = CT_tmp.copy()
            rho0_tmp = rho0_tmp.copy()
            SA_tmp[:i4] = SA_tmp[i4]
            CT_tmp[:i4] = CT_tmp[i4]
            rho0_tmp[:i4] = gsw.rho(
                SA_tmp[:i4], CT_tmp[:i4], np.zeros(i4, dtype=np.float64),
            )

        min_rho0 = float(np.min(rho0_tmp))
        diff = (min_rho0 + 0.3) - rho0_tmp
        hits = np.flatnonzero(diff > 0.0)
        if hits.size == 0:
            return float("nan")
        mlp_val = float(p_tmp[hits[-1]])
        dmlp = mlp_val - float(p_tmp.min())
        if dmlp < 20.0:
            return float("nan")

    return mlp_val
