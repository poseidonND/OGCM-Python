# OGCM_Python (portable)

Python port of `OGCM_DL.f90` — reads ocean model snapshots (HYCOM GOFS/GLBy
NetCDF, or raw RTOFS `.a/.b` archives), interpolates them onto an ADCIRC
unstructured mesh, and writes the depth-averaged baroclinic pressure
gradients, buoyancy frequencies, surface sigma-t density, mixed-layer depth
ratio, momentum dispersion, and (optionally) baroclinic sea level. Output
matches the Fortran program's NetCDF layout bit-for-bit for downstream use.

This folder is **self-contained**: drop it anywhere on the target machine,
install the pip requirements, and you can run all three entry points.

---

## 1. What's in this folder

```
OGCM_Python_portable/
├── README.md                          ← this file
├── requirements.txt                   ← pip deps
├── ogcm_dl/                           ← the Python package
│   ├── __init__.py
│   ├── constants.py                   ← Earth radius, fill values, ...
│   ├── config.py                      ← control-file parser
│   ├── io_local.py                    ← ogcm_data.txt parser
│   ├── mesh.py                        ← fort.14 reader + geometry
│   ├── interp.py                      ← bilinear HYCOM→ADCIRC weights
│   ├── ogcm_io.py                     ← HYCOM NetCDF reader
│   ├── calc_adcirc.py                 ← the physics (Calc_BC2D_*, Calc_Steric)
│   ├── output.py                      ← NetCDF writers
│   ├── runner.py                      ← main OGCM_DL loop
│   ├── regrid.py                      ← RTOFS archv → GLBy regridder
│   └── _mlp.py                        ← pure-Python gsw_mlp shim
├── scripts/                           ← command-line entry points
│   ├── run_ogcm_dl.py                 ← direct control-file runner
│   ├── run_hycom_pipeline.py          ← single .a/.b → BC2D
│   ├── run_hycom_pipeline_multi.py    ← N .a/.b → single fort.11.nc
│   │                                    (with --framework serial|parallel)
│   └── plot_output.py                 ← quick tripcolor visualization
└── examples/
    ├── fort.14                        ← example ADCIRC mesh
    └── sample_input.txt               ← example control file
```

---

## 2. Install on the target machine

Requires Python **3.10+** (uses `X | None` type hints and modern `datetime`).

```bash
cd OGCM_Python_portable
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Test the install:

```bash
python -c "import ogcm_dl, ogcm_dl.regrid, ogcm_dl.runner; print('OK')"
```

For every command below, `cd` to the `OGCM_Python_portable/` folder so that
Python picks up the local `ogcm_dl` package. The scripts prepend the folder
to `sys.path` automatically, so no editable install (`pip install -e .`) is
required.

---

## 3. Three ways to run

### 3a. Direct: you already have HYCOM GOFS/GLBy NetCDFs

Use `run_ogcm_dl.py` if you have HYCOM NetCDFs on disk (the ones with 1-D
lat/lon and 40 fixed z-levels).

```bash
python scripts/run_ogcm_dl.py \
    --control    examples/sample_input.txt \
    --ogcm-data  ogcm_data.txt \
    -v
```

Where `sample_input.txt` follows the same 8-line layout as `OGCM_DL.f90`:

```
! comment line
2024-01-01 00:00       ← start time (UTC)
2024-01-02 00:00       ← end   time (UTC)
1                      ← TMULT (multiplier on the 3-hourly OGCM cadence)
loc                    ← BCServer tag (ignored; kept for compatibility)
3                      ← OutType: 3 = TS, 4 = TS+UV, 5 = BCSL
fort.14                ← ADCIRC mesh path
bc2d_adcirc.nc         ← output NetCDF path
```

And `ogcm_data.txt` lists your local snapshots:

```
<NSNAPS>
<YYYY-MM-DD HH:MM>  <TS_FILE>  <UV_FILE>  <SSH_FILE>
...
```

### 3b. Single RTOFS archv → BC2D output

Use `run_hycom_pipeline.py` when you start from a raw RTOFS `.a` (or `.a.tar` /
`.a.gz`) and its `.b` sibling. It regrids the archive onto the GLBy grid in
memory, then runs the ADCIRC-side computation, writing one time slab.

```bash
python scripts/run_hycom_pipeline.py \
    --archv-a  ~/data/rtofs_glo.t00z.f06.archv.a.tar \
    --archv-b  ~/data/rtofs_glo.t00z.f06.archv.b \
    --grid-a   ~/data/rtofs_glo.navy_0.08.regional.grid.a \
    --fort14   examples/fort.14 \
    --outtype  3 \
    --output   bc2d_adcirc.nc \
    -v
```

Timings on a 5.4 GB tar (measured on an M-series Mac):

| Stage      |   cold  | warm .a cache |
|------------|---------|---------------|
| prep       |   80 s  |     0 s       |
| regrid     |  236 s  |   226 s       |
| ADCIRC     |   15 s  |    27 s       |
| **total**  |**5:30** | **4:13**      |

Add `--keep-intermediates` on the first run so the extracted `.a` sticks
around under `work_hycom/` and subsequent runs on the same tar hit the fast
path.

### 3c. Many RTOFS archv → single `fort.11.nc` (serial or parallel)

Use `run_hycom_pipeline_multi.py` when you have several `.a.tar/.b` pairs
and want them stacked into one output NetCDF. You pick the execution model
with `--framework`:

```
--framework serial      Process files one at a time in the main process.
                        No subprocesses, clean stack traces, easy Ctrl-C,
                        low memory pressure. Best for one file or a
                        tight-RAM machine.

--framework parallel    Farm files to a ProcessPool of --workers processes
                        (default min(cpu_count, 3)). ~4-5x wall-time
                        speedup for N files on a multi-core, ample-RAM
                        machine. Each worker holds ~7 GB peak.

--framework auto        Serial for 1 file, parallel for >1. (default)
```

Example (three snapshots, 2-way parallel):

```bash
python scripts/run_hycom_pipeline_multi.py \
    --archv-a  ~/dl/rtofs.f06.archv.a.tar \
               ~/dl/rtofs.f12.archv.a.tar \
               ~/dl/rtofs.f18.archv.a.tar \
    --archv-b  ~/dl/rtofs.f06.archv.b \
               ~/dl/rtofs.f12.archv.b \
               ~/dl/rtofs.f18.archv.b \
    --grid-a   ~/data/rtofs_glo.navy_0.08.regional.grid.a \
    --fort14   examples/fort.14 \
    --outtype  3 \
    --output   fort.11.nc \
    --framework parallel --workers 2 \
    -v
```

Same command with `--framework serial` produces bit-exact output but takes
~3x longer.

The script prints a per-file timing table at the end plus (in parallel
mode) an aggregate parallel-speedup number.

---

## 4. Plot the output

```bash
python scripts/plot_output.py --nc fort.11.nc --fort14 examples/fort.14
```

Panels all six ADCIRC variables (SigTS, BPGX, BPGY, MLD, NB, NM) as
`tripcolor` maps. Use `--field BPGX` to render a single variable. Dateline-
wrap triangles (the ones spanning ±180°) are masked so you don't get spurious
horizontal bars.

---

## 5. OutType cheat sheet

| OutType | What it produces on the ADCIRC mesh                             |
|---------|-----------------------------------------------------------------|
| **3**   | TS-only: `BPGX`, `BPGY`, `SigTS`, `MLD`, `NB`, `NM`             |
| **4**   | TS + UV: everything in OutType 3, plus `DispX`, `DispY`         |
| **5**   | Baroclinic sea level: `BCSL`                                    |

OutTypes 0/1/2 (HYCOM-grid output) are intentionally not ported.

---

## 6. Files the target machine still needs to bring

The portable folder contains all the code + a sample control file + a
sample `fort.14`. To actually run on new data you'll want on-target:

1. **HYCOM/RTOFS data** — either archv `.a/.b` files (fed to
   `run_hycom_pipeline*.py`) or the regridded GLBy-style NetCDFs (fed to
   `run_ogcm_dl.py`).
2. **RTOFS regional grid** (`rtofs_glo.navy_0.08.regional.grid.a` + `.b`)
   — only needed when starting from raw archv files.
3. **Your own ADCIRC mesh** — replace `examples/fort.14` with your own or
   pass `--fort14 /path/to/your/fort.14` at the CLI.

---

## 7. `work_hycom/` cache

The RTOFS pipeline wrappers extract each 5.4 GB `.a.tar` to an
uncompressed 12 GB `.a` under `work_hycom/` (or a per-job subdir in the
multi wrapper). Pass `--keep-intermediates` to have them cache-hit on
subsequent runs. Delete `work_hycom/` to reclaim disk.

---

## 8. Verifying the port

Every run above was cross-checked against a Fortran-generated
`fort.11.nc`. All six ADCIRC output variables agree to floating-point
precision (bit-exact when compared through the same int16 packing).
