"""Bilinear interpolation HYCOM grid -> ADCIRC mesh.

Ports `haversine`, `binarysearch`, `bl_interp`, and `Get_Interpolation_Weights`
from OGCM_DL.f90. Indices are 0-based throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEG2RAD, LAT_UL, R_EARTH


def haversine(deglon1: float, deglon2: float,
              deglat1: float, deglat2: float) -> float:
    """Great-circle distance between two points (matches Fortran haversine)."""
    deglat1 = min(LAT_UL, deglat1)
    deglat2 = min(LAT_UL, deglat2)
    dlat = DEG2RAD * (deglat2 - deglat1)
    dlon = DEG2RAD * (deglon2 - deglon1)
    lat1 = DEG2RAD * deglat1
    lat2 = DEG2RAD * deglat2
    a = (np.sin(0.5 * dlat) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R_EARTH * c


def binarysearch(array: np.ndarray, value: float, delta: float = 1e-9) -> int:
    """Largest index `i` such that `array[i] <= value` (matches Fortran binarysearch).

    Returns 0 if `value <= array[0]`. Handles ascending or descending arrays.
    Returns -1 (Fortran would return 0) when `value` is below the entire range.
    """
    length = len(array)
    if length == 0:
        return -1
    orientation = 1 if array[1] >= array[0] else -1
    left = 0
    right = length - 1
    while left <= right:
        middle = (left + right) // 2
        if abs(array[middle] - value) <= delta:
            return middle
        if orientation == 1:
            if array[middle] > value:
                right = middle - 1
            else:
                left = middle + 1
        else:
            if array[middle] < value:
                right = middle - 1
            else:
                left = middle + 1
    # `right` ends up as the largest index with array[right] <= value
    return right


def bl_interp(x_array: np.ndarray, y_array: np.ndarray,
              x: float, y: float) -> tuple[tuple[int, int, int, int],
                                            tuple[float, float, float, float]]:
    """Bilinear interpolation weights at (x, y) on a regular lat/lon grid.

    Mirrors the Fortran `bl_interp` subroutine. Returns:

      indices = (i, j, ir, jr)  # 0-based
      weights = (w1, w2, w3, w4)

    where the corners are
      (i, j),  (ir, j),  (i, jr),  (ir, jr)
    and the weights satisfy w1 + w2 + w3 + w4 = 1 except in degenerate cases.
    Distances are computed with haversine (so weights are spatially correct
    near the poles too). Handles longitudinal wrap when the input grid spans
    the full circle.
    """
    xp = len(x_array)
    yp = len(y_array)
    i = binarysearch(x_array, x)
    j = binarysearch(y_array, y)

    if i < 0:
        i = 0
        ir = 0
    else:
        ir = i + 1
        if ir >= xp:
            # Test if the grid wraps in longitude (close to 360deg span)
            if 3.0 * x_array[0] - 2.0 * x_array[1] + 360.0 < x_array[xp - 1]:
                ir = 0  # wrap
            else:
                ir = xp - 1

    x1 = x_array[i]
    x2 = x_array[ir]

    if j < 0:
        j = 0
        jr = 0
    else:
        jr = j + 1
        if jr >= yp:
            jr = yp - 1
    y1 = y_array[j]
    y2 = y_array[jr]

    if ir == i:
        x2x1 = 1.0
        x2x = 0.0
        xx1 = 1.0
    else:
        x2x1 = haversine(x1, x2, y, y)
        x2x = haversine(x, x2, y, y)
        xx1 = haversine(x1, x, y, y)
    if jr == j:
        y2y1 = 1.0
        y2y = 0.0
        yy1 = 1.0
    else:
        y2y1 = haversine(x, x, y1, y2)
        y2y = haversine(x, x, y, y2)
        yy1 = haversine(x, x, y1, y)

    denom = x2x1 * y2y1
    if denom < 1e-12:
        # Degenerate -- collapse to nearest neighbor
        return (i, j, ir, jr), (1.0, 0.0, 0.0, 0.0)

    w1 = x2x * y2y / denom
    w2 = xx1 * y2y / denom
    w3 = x2x * yy1 / denom
    w4 = xx1 * yy1 / denom
    return (i, j, ir, jr), (w1, w2, w3, w4)


@dataclass
class InterpWeights:
    """Per-node bilinear interpolation indices and weights.

    Attributes
    ----------
    indxy : np.ndarray, shape (4, NP)
        Rows: (i, j, ir, jr), 0-based indices into BC3D_Lon and BC3D_Lat.
    weights : np.ndarray, shape (4, NP)
        Bilinear weights for the 4 corners.
    indz : np.ndarray, shape (NP,)
        Largest 0-based depth index `iz` such that `BC3D_Z[iz] < dp[ip]`.
        Equal to -1 if the node is shallower than the topmost layer.
    """

    indxy: np.ndarray
    weights: np.ndarray
    indz: np.ndarray


def _haversine_vec(lon1: np.ndarray, lon2: np.ndarray,
                   lat1: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance (same formula as `haversine`)."""
    lat1c = np.minimum(LAT_UL, lat1)
    lat2c = np.minimum(LAT_UL, lat2)
    dlat = DEG2RAD * (lat2c - lat1c)
    dlon = DEG2RAD * (lon2 - lon1)
    a = (np.sin(0.5 * dlat) ** 2
         + np.cos(DEG2RAD * lat1c) * np.cos(DEG2RAD * lat2c)
         * np.sin(0.5 * dlon) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return R_EARTH * 2.0 * np.arcsin(np.sqrt(a))


def get_interpolation_weights(slam: np.ndarray, sfea: np.ndarray,
                              dp: np.ndarray,
                              bc3d_lon: np.ndarray, bc3d_lat: np.ndarray,
                              bc3d_z: np.ndarray) -> InterpWeights:
    """Build the bilinear interpolation weights for every ADCIRC node.

    Mirrors `Get_Interpolation_Weights` in the Fortran code, vectorized.
    """
    xx = np.asarray(slam, dtype=np.float64)
    if float(bc3d_lon.min()) >= 0.0:
        xx = np.where(xx < 0.0, xx + 360.0, xx)
    yy = np.asarray(sfea, dtype=np.float64)
    bb = np.asarray(dp, dtype=np.float64)

    xp = int(bc3d_lon.size)
    yp = int(bc3d_lat.size)
    i = np.searchsorted(bc3d_lon, xx, side="right") - 1
    j = np.searchsorted(bc3d_lat, yy, side="right") - 1

    wraps_lon = (3.0 * float(bc3d_lon[0]) - 2.0 * float(bc3d_lon[1]) + 360.0
                 < float(bc3d_lon[xp - 1]))

    i_neg = i < 0
    i = np.where(i_neg, 0, i)
    ir = i + 1
    ir = np.where(i_neg, 0, ir)
    ir_over = ir >= xp
    ir = np.where(ir_over, 0 if wraps_lon else xp - 1, ir)

    j_neg = j < 0
    j = np.where(j_neg, 0, j)
    jr = j + 1
    jr = np.where(j_neg, 0, jr)
    jr = np.where(jr >= yp, yp - 1, jr)

    x1 = bc3d_lon[i]
    x2 = bc3d_lon[ir]
    y1 = bc3d_lat[j]
    y2 = bc3d_lat[jr]

    same_x = ir == i
    x2x1 = np.where(same_x, 1.0, _haversine_vec(x1, x2, yy, yy))
    x2x = np.where(same_x, 0.0, _haversine_vec(xx, x2, yy, yy))
    xx1 = np.where(same_x, 1.0, _haversine_vec(x1, xx, yy, yy))
    same_y = jr == j
    y2y1 = np.where(same_y, 1.0, _haversine_vec(xx, xx, y1, y2))
    y2y = np.where(same_y, 0.0, _haversine_vec(xx, xx, yy, y2))
    yy1 = np.where(same_y, 1.0, _haversine_vec(xx, xx, y1, yy))

    denom = x2x1 * y2y1
    degenerate = denom < 1e-12
    denom_safe = np.where(degenerate, 1.0, denom)
    w1 = np.where(degenerate, 1.0, x2x * y2y / denom_safe)
    w2 = np.where(degenerate, 0.0, xx1 * y2y / denom_safe)
    w3 = np.where(degenerate, 0.0, x2x * yy1 / denom_safe)
    w4 = np.where(degenerate, 0.0, xx1 * yy1 / denom_safe)

    indxy = np.stack([i, j, ir, jr], axis=0).astype(np.int32, copy=False)
    weights = np.stack([w1, w2, w3, w4], axis=0)
    # Largest iz with z[iz] < dp  <=>  searchsorted_left(z, dp) - 1
    indz = (np.searchsorted(bc3d_z, bb, side="left") - 1).astype(np.int32)

    return InterpWeights(indxy=indxy, weights=weights, indz=indz)
