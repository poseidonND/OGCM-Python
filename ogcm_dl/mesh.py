"""ADCIRC mesh helpers.

Ports the relevant subroutines from `OGCM_DL.f90`:

  - Read_F14
  - Calc_Areas (with CAL_ELXV_SPCOOR, CAL_JAC, CAL_EDGELENGTH)
  - Calc_Derivatives (with CYLINDERMAP, COMPUTE_CYLINPROJ_SFAC, SFAC_ELEAVG)

The math/conventions (Mercator projection, Earth radius, deg<->rad swaps) are
preserved exactly. Coordinates are 0-based throughout (Fortran is 1-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import DEG2RAD, R_EARTH, SFEA0, SLAM0


@dataclass
class AdcircMesh:
    """ADCIRC unstructured mesh data.

    Attributes
    ----------
    np_ : int
        Number of nodes.
    ne : int
        Number of elements.
    slam : np.ndarray, shape (np_,)
        Longitude of each node [degrees].
    sfea : np.ndarray, shape (np_,)
        Latitude of each node [degrees].
    dp : np.ndarray, shape (np_,)
        Depth at each node [m, positive down].
    nm : np.ndarray, shape (ne, 3), int
        Connectivity (0-based node indices for each element triangle).
    areas : np.ndarray, shape (ne,)
        Area of each element in projected (Mercator) coordinates [m^2].
    total_area : np.ndarray, shape (np_,)
        Sum of element areas connected to each node [m^2].
    dphi_dx : np.ndarray, shape (ne, 3)
        Element-basis derivatives d(phi_i)/dx for i = 1, 2, 3.
    dphi_dy : np.ndarray, shape (ne, 3)
        Element-basis derivatives d(phi_i)/dy for i = 1, 2, 3.
    """

    np_: int
    ne: int
    slam: np.ndarray
    sfea: np.ndarray
    dp: np.ndarray
    nm: np.ndarray
    areas: np.ndarray = field(default_factory=lambda: np.empty(0))
    total_area: np.ndarray = field(default_factory=lambda: np.empty(0))
    dphi_dx: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    dphi_dy: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def read_f14(path: str | Path) -> AdcircMesh:
    """Reads an ADCIRC fort.14 mesh file (ports Read_F14).

    Only nodes and the element connectivity table are parsed; boundary
    information is ignored.
    """
    path = Path(path)
    with path.open("r") as fh:
        fh.readline()  # header line
        ne, np_ = map(int, fh.readline().split()[:2])

        slam = np.empty(np_, dtype=np.float64)
        sfea = np.empty(np_, dtype=np.float64)
        dp = np.empty(np_, dtype=np.float64)
        for i in range(np_):
            tok = fh.readline().split()
            slam[i] = float(tok[1])
            sfea[i] = float(tok[2])
            dp[i] = float(tok[3])

        nm = np.empty((ne, 3), dtype=np.int64)
        for i in range(ne):
            tok = fh.readline().split()
            # Convert from 1-based to 0-based
            nm[i, 0] = int(tok[2]) - 1
            nm[i, 1] = int(tok[3]) - 1
            nm[i, 2] = int(tok[4]) - 1

    return AdcircMesh(
        np_=np_, ne=ne, slam=slam, sfea=sfea, dp=dp, nm=nm,
    )


# ---------------------------------------------------------------------------
# Internal projection helpers (CAL_JAC, CAL_EDGELENGTH, CYLINDERMAP)
# ---------------------------------------------------------------------------

def _cal_jac(lon: np.ndarray, lat: np.ndarray) -> float:
    """CAL_JAC: signed Jacobian of the (lon, lat) triangle (degrees, NOT radians)."""
    xr0 = 0.5 * (lon[1] - lon[0])
    xr1 = 0.5 * (lat[1] - lat[0])
    xs0 = 0.5 * (lon[2] - lon[0])
    xs1 = 0.5 * (lat[2] - lat[0])
    return xr0 * xs1 - xr1 * xs0


def _cal_edgelength(lon: np.ndarray, lat: np.ndarray) -> float:
    """CAL_EDGELENGTH: sqrt(sum(dlx^2 + dly^2)) over the three triangle sides."""
    dlx = np.array(
        [lon[1] - lon[0], lon[2] - lon[1], lon[0] - lon[2]], dtype=np.float64
    )
    dly = np.array(
        [lat[1] - lat[0], lat[2] - lat[1], lat[0] - lat[2]], dtype=np.float64
    )
    return float(np.sqrt((dlx * dlx + dly * dly).sum()))


def _cylindermap(rlambda: float, phi: float,
                 rlambda0: float, phi0: float) -> tuple[float, float]:
    """CYLINDERMAP (Mercator). Inputs in radians; output in projected meters."""
    x = R_EARTH * (rlambda - rlambda0) * np.cos(phi0)
    y = R_EARTH * np.log(np.tan(phi) + 1.0 / np.cos(phi)) * np.cos(phi0)
    return x, y


def _cal_elxv_spcoor(lonve: np.ndarray, latve: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """CAL_ELXV_SPCOOR: project triangle nodes to Mercator coordinates.

    Inputs are in degrees. Returns (xve, yve) in projected metres (the local
    Mercator projection used by ADCIRC). Reproduces the wrap handling in the
    Fortran code for elements that straddle the 0/360 longitude seam.
    """
    lonm = np.mod(lonve, 360.0).copy()
    latm = latve.copy()
    lontmp = lonm.copy()

    jac1 = _cal_jac(lonm, latm)
    dled1 = _cal_edgelength(lonm, latm)

    if jac1 < 0.0 or (jac1 > 0.0 and dled1 > 360.0):
        # Element wraps the seam: try moving a single offending node by 360
        if int(np.sum(lonm > 180.0)) == 1:
            idx = int(np.flatnonzero(lonm > 180.0)[0])
            lonm[idx] -= 360.0
        if int(np.sum(lonm < 180.0)) == 1:
            idx = int(np.flatnonzero(lonm < 180.0)[0])
            lonm[idx] += 360.0

        jac2 = _cal_jac(lonm, latm)
        dled2 = _cal_edgelength(lonm, latm)
        if jac2 < 0.0 or (jac1 > 0.0 and dled2 > dled1):
            lonm = lontmp.copy()

    lonm_rad = lonm * DEG2RAD
    latm_rad = latm * DEG2RAD
    xve = np.empty(3, dtype=np.float64)
    yve = np.empty(3, dtype=np.float64)
    for ii in range(3):
        xve[ii], yve[ii] = _cylindermap(
            lonm_rad[ii], latm_rad[ii], SLAM0, SFEA0
        )
    return xve, yve


# ---------------------------------------------------------------------------
# Calc_Areas / Calc_Derivatives
# ---------------------------------------------------------------------------

def calc_areas(mesh: AdcircMesh) -> np.ndarray:
    """Calc_Areas: element areas + total area connected to each node.

    Mutates `mesh.nm` to flip nodes for elements with negative Jacobian (so
    every element ends up CCW after wrap handling), then fills `mesh.areas`,
    `mesh.total_area`, and an internal `_fdxe`, `_fdye` used by
    `calc_derivatives`. Returns the element areas array for convenience.
    """
    ne = mesh.ne
    nm = mesh.nm
    slam = mesh.slam
    sfea = mesh.sfea

    areas = np.zeros(ne, dtype=np.float64)
    total_area = np.zeros(mesh.np_, dtype=np.float64)
    fdxe = np.zeros((ne, 3), dtype=np.float64)
    fdye = np.zeros((ne, 3), dtype=np.float64)

    for ii in range(ne):
        lonve = np.array(
            [slam[nm[ii, 0]], slam[nm[ii, 1]], slam[nm[ii, 2]]],
            dtype=np.float64,
        )
        latve = np.array(
            [sfea[nm[ii, 0]], sfea[nm[ii, 1]], sfea[nm[ii, 2]]],
            dtype=np.float64,
        )
        lonve_check = np.mod(lonve, 360.0)
        jac1 = _cal_jac(lonve_check, latve)
        dled1 = _cal_edgelength(lonve_check, latve)
        if jac1 < 0.0 and dled1 < 360.0:
            tmp = nm[ii, :].copy()
            nm[ii, 0] = tmp[2]
            nm[ii, 1] = tmp[1]
            nm[ii, 2] = tmp[0]

        lonve = np.array(
            [slam[nm[ii, 0]], slam[nm[ii, 1]], slam[nm[ii, 2]]],
            dtype=np.float64,
        )
        latve = np.array(
            [sfea[nm[ii, 0]], sfea[nm[ii, 1]], sfea[nm[ii, 2]]],
            dtype=np.float64,
        )
        xve, yve = _cal_elxv_spcoor(lonve, latve)
        x1, x2, x3 = xve
        y1, y2, y3 = yve
        x2mx1 = x2 - x1
        x3mx2 = x3 - x2
        x1mx3 = x1 - x3
        y2my1 = y2 - y1
        y3my2 = y3 - y2
        y1my3 = y1 - y3

        fdxe[ii, 0] = -y3my2
        fdxe[ii, 1] = -y1my3
        fdxe[ii, 2] = -y2my1
        fdye[ii, 0] = x3mx2
        fdye[ii, 1] = x1mx3
        fdye[ii, 2] = x2mx1

        areas[ii] = 0.5 * (x1mx3 * (-y3my2) + x3mx2 * y1my3)
        total_area[nm[ii, 0]] += areas[ii]
        total_area[nm[ii, 1]] += areas[ii]
        total_area[nm[ii, 2]] += areas[ii]

    mesh.areas = areas
    mesh.total_area = total_area
    # Stash for calc_derivatives. Underscore-prefixed attributes are
    # implementation details; calc_derivatives clears them.
    setattr(mesh, "_fdxe", fdxe)
    setattr(mesh, "_fdye", fdye)
    return areas


def _compute_cylinproj_sfac(slam_rad: np.ndarray, sfea_rad: np.ndarray,
                            nm: np.ndarray, ne: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """COMPUTE_CYLINPROJ_SFAC: returns SFmxEle, SFmyEle for Mercator (ICS=22).

    Inputs `slam_rad`, `sfea_rad` are nodal coords in radians.
    """
    # MEXP = MOD(22, 20) = 2 for Mercator
    mexp = 2
    sfmx = np.cos(SFEA0) / np.cos(sfea_rad)
    sfmy = sfmx ** (mexp - 1)

    # Element-average (SFAC_ELEAVG): mean of nodal values over the 3 triangle
    # vertices.
    onethird = 1.0 / 3.0
    sfmx_ele = (sfmx[nm[:, 0]] + sfmx[nm[:, 1]] + sfmx[nm[:, 2]]) * onethird
    sfmy_ele = (sfmy[nm[:, 0]] + sfmy[nm[:, 1]] + sfmy[nm[:, 2]]) * onethird
    return sfmx_ele, sfmy_ele


def calc_derivatives(mesh: AdcircMesh) -> None:
    """Calc_Derivatives: fills mesh.dphi_dx and mesh.dphi_dy.

    Must be called after `calc_areas`.
    """
    if not hasattr(mesh, "_fdxe"):
        raise RuntimeError("calc_derivatives requires calc_areas to have run.")

    fdxe = getattr(mesh, "_fdxe")
    fdye = getattr(mesh, "_fdye")

    slam_rad = mesh.slam * DEG2RAD
    sfea_rad = mesh.sfea * DEG2RAD
    sfmx_ele, sfmy_ele = _compute_cylinproj_sfac(
        slam_rad, sfea_rad, mesh.nm, mesh.ne
    )

    area_ie2 = 2.0 * mesh.areas
    # Avoid division by zero in degenerate elements; fall back to 0 derivative.
    safe = area_ie2 != 0.0
    dphi_dx = np.zeros((mesh.ne, 3), dtype=np.float64)
    dphi_dy = np.zeros((mesh.ne, 3), dtype=np.float64)
    for k in range(3):
        dphi_dx[safe, k] = fdxe[safe, k] * sfmx_ele[safe] / area_ie2[safe]
        dphi_dy[safe, k] = fdye[safe, k] * sfmy_ele[safe] / area_ie2[safe]

    mesh.dphi_dx = dphi_dx
    mesh.dphi_dy = dphi_dy
    # Clean up scratch
    delattr(mesh, "_fdxe")
    delattr(mesh, "_fdye")
