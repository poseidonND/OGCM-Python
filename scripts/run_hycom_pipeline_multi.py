#!/usr/bin/env python3
"""Parallel HYCOM archv -> single BC2D output pipeline.

Given N pairs of RTOFS `.a.tar` / `.b` files, this script:

  1. Farms each file out to its own worker process. Each worker:
       - extracts / stages the `.a` next to its `.b`,
       - regrids the archv to the GLBy grid in memory,
       - runs the ADCIRC-side OGCM_DL computation for that single time,
       - returns the resulting numpy arrays over a Pipe.
  2. Collects all worker results, sorts them by timestamp, and writes them
     into a single output NetCDF (default `fort.11.nc`) with one time slab
     per input file.

Choice of execution framework:

  --framework serial      Process files one at a time in the main process.
                          Cleanest stack traces, easiest Ctrl-C, no worker
                          spawn cost. Best for 1 file, or memory-tight
                          machines.
  --framework parallel    Farm files out to a ProcessPool of --workers.
                          Best for many files on multi-core, ample-RAM
                          machines. Each worker holds ~7 GB peak, so
                          worker count is memory-bound: default is
                          min(cpu_count, 3); override with --workers N.
  --framework auto        (default) Serial for 1 file, parallel for >1.

Example (three snapshots, running two at a time in parallel):

    python scripts/run_hycom_pipeline_multi.py \\
        --archv-a  ~/dl/rtofs.f06.archv.a.tar \\
                   ~/dl/rtofs.f12.archv.a.tar \\
                   ~/dl/rtofs.f18.archv.a.tar \\
        --archv-b  ~/dl/rtofs.f06.archv.b \\
                   ~/dl/rtofs.f12.archv.b \\
                   ~/dl/rtofs.f18.archv.b \\
        --grid-a   ~/Desktop/RTOFS_Post/rtofs_glo.navy_0.08.regional.grid.a \\
        --fort14   fort.14 \\
        --outtype  3 \\
        --output   fort.11.nc \\
        --framework parallel \\
        --workers  2

Same command with `--framework serial` processes the three files one at a
time (about 3x the wall time of `--framework parallel --workers 3`).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hycom_pipeline import _prepare_archv  # noqa: E402
from ogcm_dl.config import Config  # noqa: E402
from ogcm_dl.mesh import calc_areas, calc_derivatives, read_f14  # noqa: E402
from ogcm_dl.output import (  # noqa: E402
    init_netcdf_adc, init_netcdf_bcsl,
    update_netcdf_bcsl, update_netcdf_ts,
)
from ogcm_dl.regrid import convert_archv_to_glby  # noqa: E402
from ogcm_dl.runner import compute_bc2d_from_grid  # noqa: E402

log = logging.getLogger("hycom_pipeline_multi")


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

@dataclass
class WorkerJob:
    """Everything a worker needs to process one .a/.b pair standalone."""
    idx: int
    archv_a: Path
    archv_b: Path
    work_subdir: Path
    grid_a: Path
    fort14: Path
    out_type: int
    keep_intermediates: bool


def _init_worker(log_level: int) -> None:
    """Runs in each worker process at import time.

    Tames numpy/BLAS thread pools so that N workers x M BLAS threads doesn't
    oversubscribe the machine.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
        force=True,
    )


def _process_one(job: WorkerJob) -> dict[str, Any]:
    """Full per-file pipeline: prepare .a, regrid, compute ADCIRC result.

    Runs entirely inside one worker process. Returns a small picklable dict
    the parent can collect and combine.
    """
    t0 = time.perf_counter()
    log.info("[job %d] start: %s", job.idx, job.archv_a.name)

    job.work_subdir.mkdir(parents=True, exist_ok=True)
    dest_a = _prepare_archv(job.archv_a, job.archv_b, job.work_subdir)
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = convert_archv_to_glby(
        archv_a=dest_a,
        grid_a=job.grid_a,
        out_dir=job.work_subdir,
        do_uv=(job.out_type == 4),
        return_in_memory=True,
    )
    t_regrid = time.perf_counter() - t1

    t2 = time.perf_counter()
    mesh = read_f14(job.fort14)
    calc_areas(mesh)
    calc_derivatives(mesh)
    cfg = Config(
        ts=result.when, te=result.when, tmult=1, bc_server="local",
        out_type=job.out_type, fort14=job.fort14,
        bc2d_name=Path("in-memory-multi"),
    )
    adc = compute_bc2d_from_grid(
        cfg, when=result.when,
        ts_grid=result.ts_grid, uv_grid=result.uv_grid,
        mesh=mesh,
    )
    t_adcirc = time.perf_counter() - t2

    if not job.keep_intermediates:
        if dest_a.is_symlink():
            dest_a.unlink()
        elif dest_a.exists():
            dest_a.unlink()
        marker = dest_a.with_suffix(".a.ready")
        if marker.exists():
            marker.unlink()

    t_total = time.perf_counter() - t0
    log.info(
        "[job %d] done in %.1f s (prep=%.1f regrid=%.1f adcirc=%.1f)",
        job.idx, t_total, t_prep, t_regrid, t_adcirc,
    )
    adc["_timings"] = {
        "prep": t_prep, "regrid": t_regrid, "adcirc": t_adcirc,
        "total": t_total, "idx": job.idx,
    }
    return adc


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

def _write_combined(out_path: Path, out_type: int, fort14: Path,
                    slabs: list[dict[str, Any]]) -> None:
    """Write all worker slabs, in chronological order, to a single NetCDF."""
    mesh = read_f14(fort14)
    calc_areas(mesh)
    calc_derivatives(mesh)

    if out_path.exists():
        log.warning("Output file %s already exists; overwriting", out_path)
        out_path.unlink()

    if out_type in (3, 4):
        init_netcdf_adc(out_path, mesh, out_type)
    elif out_type == 5:
        init_netcdf_bcsl(out_path, mesh)
    else:
        raise ValueError(f"Unsupported out_type: {out_type}")

    slabs_sorted = sorted(slabs, key=lambda s: s["when"])
    seen: set[datetime] = set()
    for s in slabs_sorted:
        w = s["when"]
        if w in seen:
            raise RuntimeError(
                f"Two workers returned results for the same timestamp {w!r}; "
                f"input pairs likely have duplicates."
            )
        seen.add(w)

    for time_index, s in enumerate(slabs_sorted):
        when = s["when"]
        log.info("Writing slab %d: %s", time_index, when.isoformat(" "))
        if out_type in (3, 4):
            update_netcdf_ts(out_path, time_index, when,
                             s["ts"], s.get("uv"))
        else:
            update_netcdf_bcsl(out_path, time_index, when, s["bcsl"])

    log.info("Wrote %d slabs to %s", len(slabs_sorted), out_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--archv-a", type=Path, nargs="+", required=True,
                   help="One or more archv .a (or .a.tar / .a.gz) files.")
    p.add_argument("--archv-b", type=Path, nargs="+", required=True,
                   help="Matching .b files (same order and count as --archv-a).")
    p.add_argument("--grid-a", type=Path, required=True,
                   help="Regional grid .a file (same one used for all inputs).")
    p.add_argument("--fort14", type=Path, required=True,
                   help="ADCIRC mesh (fort.14).")
    p.add_argument("--outtype", type=int, default=3, choices=(3, 4, 5),
                   help="OGCM_DL OutType (3=TS, 4=TS+UV, 5=BCSL).")
    p.add_argument("--output", type=Path, default=Path("fort.11.nc"),
                   help="Combined output NetCDF (default fort.11.nc).")
    p.add_argument("--work-dir", type=Path, default=Path("work_hycom"),
                   help="Root for per-job scratch subdirectories.")
    p.add_argument("--framework", choices=("serial", "parallel", "auto"),
                   default="auto",
                   help="Execution framework. 'serial' processes files one "
                        "at a time in the main process (no subprocesses, "
                        "easy Ctrl-C, easier stack traces). 'parallel' uses "
                        "a ProcessPool of --workers. 'auto' (default) picks "
                        "serial when only one file is given, parallel "
                        "otherwise.")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel worker processes (only used "
                        "when --framework=parallel). Default: "
                        "min(cpu_count, 3). Memory-bound; ~7 GB per worker.")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep each worker's extracted .a after processing.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v INFO, -vv DEBUG.")
    return p.parse_args()


def _pair_and_validate(a_paths: list[Path], b_paths: list[Path]
                       ) -> list[tuple[Path, Path]]:
    """Match --archv-a[i] with --archv-b[i], validating both exist."""
    if len(a_paths) != len(b_paths):
        raise SystemExit(
            f"--archv-a has {len(a_paths)} entries but --archv-b has "
            f"{len(b_paths)}; they must match 1:1."
        )
    pairs = []
    for a, b in zip(a_paths, b_paths):
        a = a.expanduser().resolve()
        b = b.expanduser().resolve()
        if not a.exists():
            raise SystemExit(f"Missing --archv-a file: {a}")
        if not b.exists():
            raise SystemExit(f"Missing --archv-b file: {b}")
        pairs.append((a, b))
    return pairs


def main() -> int:
    args = _parse_args()

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
    )

    pairs = _pair_and_validate(args.archv_a, args.archv_b)
    n_files = len(pairs)

    # Decide serial vs parallel. `auto` treats a single-file run as serial
    # (no ProcessPool overhead, cleaner stack traces) and multi-file as
    # parallel with the default worker count.
    framework = args.framework
    if framework == "auto":
        framework = "serial" if n_files <= 1 else "parallel"

    if framework == "serial":
        n_workers = 1
        if args.workers is not None and args.workers != 1:
            log.warning(
                "--workers=%d ignored because --framework=serial",
                args.workers,
            )
    else:  # parallel
        n_workers = args.workers or min(mp.cpu_count(), 3)
        n_workers = max(1, min(n_workers, n_files))
    log.info("Processing %d file(s) [%s framework, n_workers=%d]",
             n_files, framework, n_workers)

    work_root = args.work_dir.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    jobs: list[WorkerJob] = []
    for i, (a, b) in enumerate(pairs):
        jobs.append(WorkerJob(
            idx=i,
            archv_a=a,
            archv_b=b,
            work_subdir=work_root / f"job_{i:03d}_{a.stem}",
            grid_a=args.grid_a.expanduser().resolve(),
            fort14=args.fort14.expanduser().resolve(),
            out_type=args.outtype,
            keep_intermediates=args.keep_intermediates,
        ))

    t_total0 = time.perf_counter()

    slabs: list[dict[str, Any]] = []
    if framework == "serial":
        # Run everything in the main process, sequentially. Keeps stack
        # traces flat, lets the user Ctrl-C mid-file, and avoids the
        # 1-2 s per-worker Python spawn cost.
        for job in jobs:
            slabs.append(_process_one(job))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(level,),
        ) as pool:
            fut_to_idx = {pool.submit(_process_one, job): job.idx
                          for job in jobs}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    slab = fut.result()
                except Exception:
                    log.exception("Job %d failed", idx)
                    raise
                slabs.append(slab)

    t_workers = time.perf_counter() - t_total0

    t0 = time.perf_counter()
    _write_combined(args.output.resolve(), args.outtype,
                    args.fort14.expanduser().resolve(), slabs)
    t_combine = time.perf_counter() - t0
    t_total = time.perf_counter() - t_total0

    print("\n=== Multi-file pipeline timings ===", file=sys.stderr)
    per_job = sorted((s["_timings"] for s in slabs),
                     key=lambda t: t["idx"])
    for tj in per_job:
        print(
            f"  job {tj['idx']:3d}  prep={tj['prep']:6.1f}s  "
            f"regrid={tj['regrid']:6.1f}s  "
            f"adcirc={tj['adcirc']:6.1f}s  "
            f"total={tj['total']:6.1f}s",
            file=sys.stderr,
        )
    print(f"  workers (wall)   : {t_workers:7.2f} s  "
          f"({framework}, n_workers={n_workers})",
          file=sys.stderr)
    print(f"  combine + write  : {t_combine:7.2f} s",
          file=sys.stderr)
    print(f"  total (wall)     : {t_total:7.2f} s  "
          f"for {n_files} files",
          file=sys.stderr)
    if framework == "parallel" and n_workers > 1:
        speedup = (sum(t["total"] for t in per_job) / t_workers
                   if t_workers > 0 else 1.0)
        print(f"  parallel speedup : {speedup:.2f}x  "
              f"(vs summed per-job times)", file=sys.stderr)

    log.info("Done. Output: %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
