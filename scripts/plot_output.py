#!/usr/bin/env python3
"""Plot the bc2d_adcirc.nc output on the ADCIRC mesh.

Reads `x`, `y`, `element` and the six TS-side fields (BPGX, BPGY, SigTS, MLD,
NB, NM) and renders one panel per field via matplotlib's tripcolor over the
unstructured mesh. The output is saved as a single PNG.

Usage:
    python scripts/plot_output.py [--output bc2d_adcirc.nc] [--time 0]
                                  [--png bc2d_adcirc.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from netCDF4 import Dataset


def _slice_along_time(v: np.ndarray, time_index: int, dims: tuple[str, ...],
                      ) -> np.ndarray:
    """Extract `v[..., time_index, ...]` regardless of which axis is `time`."""
    if "time" not in dims:
        return v
    ax = dims.index("time")
    return np.take(v, time_index, axis=ax)


def _load(path: Path, time_index: int) -> dict:
    with Dataset(str(path), mode="r") as nc:
        x = np.asarray(nc.variables["x"][:], dtype=np.float64)
        y = np.asarray(nc.variables["y"][:], dtype=np.float64)
        elem_dims = nc.variables["element"].dimensions
        element = np.asarray(nc.variables["element"][:], dtype=np.int64)
        # Make sure the element table is shaped (nele, nvertex) regardless of
        # which dim ordering the writer used.
        if elem_dims == ("nvertex", "nele"):
            element = element.T
        element = element - 1

        time_var = nc.variables.get("time")
        if time_var is None:
            time_str = ""
        else:
            arr = np.asarray(time_var[:])
            if arr.dtype.kind in ("S", "U"):
                if arr.ndim == 2:
                    col = (
                        arr[time_index, :] if time_var.dimensions[0] == "time"
                        else arr[:, time_index]
                    )
                else:
                    col = arr
                time_str = b"".join(
                    b"" if x is np.ma.masked else bytes(x) for x in col
                ).decode("ascii", "ignore")
            else:
                time_str = str(arr.flat[time_index])

        fields = {}
        for name in ("BPGX", "BPGY", "SigTS", "MLD", "NB", "NM"):
            if name in nc.variables:
                var = nc.variables[name]
                v = np.asarray(var[:], dtype=np.float64)
                fields[name] = _slice_along_time(v, time_index, var.dimensions)

    return {
        "x": x,
        "y": y,
        "element": element,
        "fields": fields,
        "time_str": time_str.strip(),
    }


def _symlog_lim(values: np.ndarray, pct: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(np.abs(finite), pct))


def _plot_panel(ax, tri: Triangulation, values: np.ndarray, title: str,
                cmap: str, diverging: bool = False) -> None:
    if diverging:
        lim = _symlog_lim(values)
        if lim == 0:
            lim = 1.0
        m = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                         vmin=-lim, vmax=lim)
    else:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99))
            if vmin == vmax:
                vmax = vmin + 1.0
        m = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                         vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", type=Path, default=Path("bc2d_adcirc.nc"))
    p.add_argument("--png", type=Path, default=Path("bc2d_adcirc.png"))
    p.add_argument("--time", "-t", type=int, default=0,
                   help="Time index (default: 0).")
    p.add_argument("--field", "-f", type=str, default=None,
                   choices=["BPGX", "BPGY", "SigTS", "MLD", "NB", "NM"],
                   help="If given, plot only this field as a single panel.")
    args = p.parse_args()

    if not args.output.exists():
        raise SystemExit(f"Output file {args.output} not found.")

    data = _load(args.output, args.time)
    tri = Triangulation(data["x"], data["y"], data["element"])

    # Mask out any triangle whose longitude span is unrealistically large.
    # A global ADCIRC mesh wraps across the +/-180 seam, so triangles that
    # share nodes across the seam draw as huge horizontal bars in tripcolor.
    # Anything spanning more than 180 deg cannot be a real element.
    elem = data["element"]
    x = data["x"]
    seam_span = (
        np.max(x[elem], axis=1) - np.min(x[elem], axis=1)
    )
    seam_mask = seam_span > 180.0
    if seam_mask.any():
        tri.set_mask(seam_mask)
        print(f"masked {seam_mask.sum()} dateline-wrapping elements")

    panels = [
        ("SigTS", "Surface sigma-t  [kg/m^3 - 1000]", "viridis", False),
        ("MLD",   "Mixed-layer depth ratio  [-]",     "magma",   False),
        ("BPGX",  "Baroclinic PG, x  [m/s^2]",        "RdBu_r",  True),
        ("BPGY",  "Baroclinic PG, y  [m/s^2]",        "RdBu_r",  True),
        ("NB",    "Bottom buoyancy frequency  [1/s]", "cividis", False),
        ("NM",    "Mean buoyancy frequency  [1/s]",   "cividis", False),
    ]

    title = "OGCM_Python BC2D ADCIRC output"
    if data["time_str"]:
        title += f"  --  {data['time_str']}"

    if args.field is not None:
        spec = next(p for p in panels if p[0] == args.field)
        if spec[0] not in data["fields"]:
            raise SystemExit(f"Field {spec[0]!r} not present in {args.output}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        fig.suptitle(title, fontsize=13)
        _plot_panel(ax, tri, data["fields"][spec[0]], spec[1], spec[2], spec[3])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=13)
        for ax, (name, label, cmap, diverging) in zip(axes.ravel(), panels):
            if name not in data["fields"]:
                ax.set_visible(False)
                continue
            _plot_panel(ax, tri, data["fields"][name], label, cmap, diverging)

    fig.savefig(args.png, dpi=150)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
