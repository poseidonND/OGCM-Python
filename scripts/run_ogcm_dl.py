#!/usr/bin/env python3
"""CLI entry point. Replaces PROGRAM OGCM_DL.

Usage:
    python scripts/run_ogcm_dl.py --control <file>
    python scripts/run_ogcm_dl.py < <file>

The control file format and behavior mirror `Read_Input_File()` from the
Fortran program (see README.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script directly from the repo root without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogcm_dl.calc_adcirc import NP_CHUNK_DEFAULT  # noqa: E402
from ogcm_dl.config import read_control_file  # noqa: E402
from ogcm_dl.runner import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Python port of OGCM_DL.f90 (ADCIRC-grid OutType 3/4/5).",
    )
    parser.add_argument(
        "--control", "-c", type=Path, default=None,
        help=("Control file path. If omitted, the control file is read from "
              "stdin (matching the original Fortran program).")
    )
    parser.add_argument(
        "--ogcm-data", type=Path, default=Path("ogcm_data.txt"),
        help="Path to ogcm_data.txt (default: ./ogcm_data.txt).",
    )
    parser.add_argument(
        "--np-chunk", type=int, default=None,
        help="Number of ADCIRC nodes processed per chunk in the "
             "TS/UV/BCSL calculators. Lower this on tight-RAM machines "
             "with large fort.14 meshes (default: 100000).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.control is not None:
        cfg = read_control_file(args.control)
    else:
        if sys.stdin.isatty():
            parser.error(
                "No control file given. Pass --control <file> or pipe one in."
            )
        cfg = read_control_file(sys.stdin)

    np_chunk = args.np_chunk if args.np_chunk is not None else NP_CHUNK_DEFAULT
    run(cfg, ogcm_data_path=args.ogcm_data, np_chunk=np_chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
