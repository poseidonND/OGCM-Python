#!/usr/bin/env python3
"""End-to-end pipeline: HYCOM archv (.a/.b) -> ADCIRC BC2D output.

Given RTOFS-style archv files (`*.archv.a`, `*.archv.b`, optionally the
`*.a.tar` variant) plus a regional grid file, this script:

  1. Extracts the .a from its tar, if needed.
  2. Regrids the archv from the tripolar Navy 0.08 grid + 41 hybrid layers
     onto the GLBy standard grid (regular 1-D lat/lon, 40 z-levels).
     Produces one or two GLBy-style NetCDFs (`*_ts3z.nc`, and `*_uv3z.nc`
     when --outtype 4).
  3. Writes an `ogcm_data.txt` index that points at the produced NetCDFs.
  4. Writes a control file matching the archive's timestamp.
  5. Runs `run_ogcm_dl.py` on the pair to produce the ADCIRC BC2D output.

Example:
    python scripts/run_hycom_pipeline.py \\
        --archv-a ~/Downloads/rtofs_glo.t00z.f06.archv.a.tar \\
        --archv-b ~/Downloads/rtofs_glo.t00z.f06.archv.b \\
        --grid-a  ~/Desktop/RTOFS_Post/rtofs_glo.navy_0.08.regional.grid.a \\
        --fort14  fort.14 \\
        --outtype 3 \\
        --output  bc2d_adcirc.nc
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ogcm_dl.regrid import convert_archv_to_glby  # noqa: E402
from ogcm_dl.config import read_control_file  # noqa: E402
from ogcm_dl.runner import run_single_step  # noqa: E402

log = logging.getLogger("hycom_pipeline")


# --------------------------------------------------------------------------
# Preparing the archive .a (extraction from tar / co-locating with .b)
# --------------------------------------------------------------------------

def _prepare_archv(archv_a_arg: Path, archv_b_arg: Path, work_dir: Path
                   ) -> Path:
    """Resolve the incoming .a/.b pair into a co-located .a on disk.

    Handles three cases transparently:

      * `.a.tar` archive containing a single `.a` file (RTOFS distribution).
      * `.a.gz` compressed file.
      * A plain `.a` file already on disk.

    In all cases the resulting `.a` is placed next to (or same-stem as) the
    `.b` file so utils in ogcm_dl.regrid can find the .b sibling naturally.
    """
    archv_a_arg = archv_a_arg.expanduser().resolve()
    archv_b_arg = archv_b_arg.expanduser().resolve()

    if not archv_b_arg.exists():
        raise FileNotFoundError(f"Missing .b file: {archv_b_arg}")

    # Target .a lives next to the .b (same stem) inside work_dir.
    stem = archv_b_arg.stem   # e.g. rtofs_glo.t00z.f06.archv
    dest_a = work_dir / f"{stem}.a"
    dest_b = work_dir / f"{stem}.b"

    # Copy the .b into work_dir (unmodified). We keep the .b's own timestamp.
    if dest_b.resolve() != archv_b_arg.resolve():
        log.info("Staging .b -> %s", dest_b)
        shutil.copy2(archv_b_arg, dest_b)

    name = archv_a_arg.name.lower()
    if name.endswith(".a.tar"):
        # Fast cache check first: if a .a already sits in work_dir, its
        # mtime is at least as recent as the tar's, and it's >= 1 GB
        # (i.e. plausibly a full HYCOM archive), reuse it. This avoids
        # having to open the gzip'd tar just to read a member size --
        # `tarfile.open()` on a 5.4 GB .tgz has to scan the entire stream
        # (~35 s cost) and dominates the "prepare" step.
        marker = dest_a.with_suffix(".a.ready")
        if (dest_a.exists() and marker.exists()
                and marker.read_text().strip() == str(archv_a_arg.resolve())
                and dest_a.stat().st_mtime
                    >= archv_a_arg.stat().st_mtime):
            log.info("Reusing already-extracted %s (cache hit)", dest_a)
            return dest_a

        log.info("Extracting %s -> %s", archv_a_arg, dest_a)
        with tarfile.open(archv_a_arg, "r") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and m.name.endswith(".a")]
            if not members:
                raise RuntimeError(
                    f"No .a file found inside {archv_a_arg}. "
                    f"Contents: {[m.name for m in tar.getmembers()]}"
                )
            if len(members) > 1:
                log.warning("Multiple .a files in tar; using the first: %s",
                            members[0].name)
            m = members[0]
            with tar.extractfile(m) as src:
                with open(dest_a, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        # Drop a marker sidecar so a future run can trust the cache without
        # re-opening the gzip'd tar.
        marker.write_text(str(archv_a_arg.resolve()))
        return dest_a

    if name.endswith(".a.gz"):
        import gzip
        log.info("Decompressing %s -> %s", archv_a_arg, dest_a)
        with gzip.open(archv_a_arg, "rb") as src, open(dest_a, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        return dest_a

    if name.endswith(".a"):
        # Just link it into work_dir so its .b sibling is next to it.
        if dest_a.resolve() != archv_a_arg.resolve():
            log.info("Linking .a -> %s", dest_a)
            if dest_a.exists() or dest_a.is_symlink():
                dest_a.unlink()
            try:
                dest_a.symlink_to(archv_a_arg)
            except OSError:
                shutil.copy2(archv_a_arg, dest_a)
        return dest_a

    raise ValueError(f"Unrecognized .a extension: {archv_a_arg.name}")


# --------------------------------------------------------------------------
# Writing the OGCM index + control file
# --------------------------------------------------------------------------

def _write_ogcm_data(path: Path, when: datetime, ts3z: Path,
                     uv3z: Path | None) -> None:
    ts_name = str(ts3z)
    uv_name = str(uv3z) if uv3z is not None else "none"
    ssh_name = "none"  # unused for outtype 3/4
    with path.open("w") as fh:
        fh.write("1\n")
        fh.write(f"{when:%Y-%m-%d %H:%M} {ts_name} {uv_name} {ssh_name}\n")
    log.info("Wrote OGCM index: %s", path)


def _write_control(path: Path, when: datetime, outtype: int, fort14: Path,
                   bc2d_out: Path, tmult: int) -> None:
    with path.open("w") as fh:
        fh.write("! Auto-generated by run_hycom_pipeline.py\n")
        fh.write(f"{when:%Y-%m-%d %H:%M}\n")
        fh.write(f"{when:%Y-%m-%d %H:%M}\n")
        fh.write(f"{tmult}\n")
        fh.write("local\n")
        fh.write(f"{outtype}\n")
        fh.write(f"{fort14}\n")
        fh.write(f"{bc2d_out}\n")
    log.info("Wrote control file: %s", path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--archv-a", type=Path, required=True,
                   help="HYCOM archive .a (may be .a, .a.tar, or .a.gz).")
    p.add_argument("--archv-b", type=Path, required=True,
                   help="Matching HYCOM archive .b file.")
    p.add_argument("--grid-a", type=Path, required=True,
                   help="Regional grid .a file (with matching .b) that "
                        "defines the source plat/plon curvilinear coords.")
    p.add_argument("--fort14", type=Path, default=Path("fort.14"),
                   help="ADCIRC mesh file (default: fort.14).")
    p.add_argument("--work-dir", type=Path, default=Path("work_hycom"),
                   help="Directory for intermediate files (default: ./work_hycom).")
    p.add_argument("--output", "-o", type=Path, default=Path("bc2d_adcirc.nc"),
                   help="Final BC2D NetCDF output (default: bc2d_adcirc.nc).")
    p.add_argument("--outtype", type=int, choices=(3, 4, 5), default=3,
                   help="OGCM_DL OutType: 3=TS-only, 4=TS+UV, 5=BCSL (default 3).")
    p.add_argument("--tmult", type=int, default=1,
                   help="TMULT for the control file (default 1). With a "
                        "single archive this only affects the loop cadence; "
                        "the run does one step regardless.")
    p.add_argument("--no-run", action="store_true",
                   help="Skip running run_ogcm_dl.py; just produce the "
                        "regridded GLBy-style NetCDF and ogcm_data.txt.")
    p.add_argument("--via-file", action="store_true",
                   help="Force the wrapper to read the regridded GLBy NC "
                        "back from disk (the old default). By default we "
                        "keep the regridded grid in memory and hand it "
                        "straight to run_ogcm_dl, saving ~90 s per file.")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Do not delete the extracted .a after regridding.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v INFO, -vv DEBUG.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_total0 = time.perf_counter()

    # ---- 1. Prepare .a/.b in work_dir --------------------------------------
    t0 = time.perf_counter()
    archv_a = _prepare_archv(args.archv_a, args.archv_b, work_dir)
    timings["1_prepare_archv"] = time.perf_counter() - t0

    # ---- 2. Regrid to GLBy-style NC(s) -------------------------------------
    want_uv = args.outtype == 4
    keep_in_memory = not args.via_file
    log.info("=== Regridding archv -> GLBy-style NetCDF%s ===",
             " (also kept in memory)" if keep_in_memory else "")
    t0 = time.perf_counter()
    result = convert_archv_to_glby(
        archv_a=archv_a,
        grid_a=args.grid_a.expanduser().resolve(),
        out_dir=work_dir,
        do_uv=want_uv,
        return_in_memory=keep_in_memory and not args.no_run,
    )
    timings["2_regrid"] = time.perf_counter() - t0
    if args.via_file:
        # Only announce the intermediate NC when we actually wrote it in
        # this run (i.e. --via-file). On the in-memory fast path the
        # ts3z/uv3z paths in `result` are notional -- the file may be
        # missing or stale from an earlier run, so don't advertise it.
        if result.ts3z.exists():
            log.info("Wrote ts3z: %s", result.ts3z)
        if result.uv3z is not None and result.uv3z.exists():
            log.info("Wrote uv3z: %s", result.uv3z)

    # ---- 3. Write ogcm_data.txt and control file ---------------------------
    ogcm_data_path = ROOT / "ogcm_data.txt"
    control_path = ROOT / "sample_input.txt"
    # ogcm_data.txt is only meaningful when the ts3z / uv3z NCs actually
    # exist on disk (i.e. the --via-file path). On the in-memory fast path
    # we still write control_path (used by run_single_step for out_type,
    # fort14, bc2d_name), but skip the stale ogcm_data.txt.
    if args.via_file or args.no_run:
        _write_ogcm_data(ogcm_data_path, result.when,
                         result.ts3z, result.uv3z)
    _write_control(
        control_path, result.when, args.outtype,
        args.fort14.resolve(), args.output.resolve(), args.tmult,
    )

    if not args.keep_intermediates:
        # Only remove the extracted / linked .a; leave the regridded NC in
        # place (it's the actual pipeline output the user cares about).
        if archv_a.is_symlink():
            archv_a.unlink()
        elif archv_a.exists() and archv_a.parent == work_dir:
            log.info("Removing extracted archive %s", archv_a)
            archv_a.unlink()

    # ---- 4. Run run_ogcm_dl -------------------------------------------------
    if args.no_run:
        log.info("Skipping run_ogcm_dl (per --no-run).")
        return 0

    if keep_in_memory:
        # Fast path: reuse the in-memory grids the regrid step already
        # produced, avoiding the ~90 s cost of writing + re-reading the
        # intermediate NC.
        log.info("=== Running OGCM_DL (in-process, in-memory grids) ===")
        t0 = time.perf_counter()
        cfg = read_control_file(control_path)
        run_single_step(
            cfg, when=result.when,
            ts_grid=result.ts_grid,
            uv_grid=result.uv_grid,
        )
        timings["3_ogcm_dl"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_total0
        _log_timing_summary(timings)
        log.info("Pipeline complete. BC2D output: %s", args.output.resolve())
        return 0

    # Slow path: subprocess a fresh Python that re-reads the intermediate NC.
    # Kept behind --via-file so users can compare or debug the read path.
    runner_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ogcm_dl.py"),
        "--control", str(control_path),
        "--ogcm-data", str(ogcm_data_path),
        "-" + "v" * max(1, args.verbose),
    ]
    log.info("=== Running OGCM_DL (subprocess, via file) ===")
    log.info("Command: %s", " ".join(runner_cmd))
    t0 = time.perf_counter()
    rc = subprocess.call(runner_cmd)
    timings["3_ogcm_dl"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total0
    _log_timing_summary(timings)
    if rc != 0:
        log.error("run_ogcm_dl.py exited with code %d", rc)
        return rc

    log.info("Pipeline complete. BC2D output: %s", args.output.resolve())
    return 0


def _log_timing_summary(timings: dict[str, float]) -> None:
    """Print a compact per-stage timing summary at the end of the run."""
    total = timings.get("total", sum(v for k, v in timings.items()
                                     if k != "total"))
    print("\n=== Pipeline timings ===", file=sys.stderr)
    for k in sorted(timings):
        if k == "total":
            continue
        pct = 100.0 * timings[k] / total if total > 0 else 0.0
        print(f"  {k:20s} {timings[k]:7.2f} s  ({pct:5.1f} %)",
              file=sys.stderr)
    print(f"  {'total':20s} {total:7.2f} s",
          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
