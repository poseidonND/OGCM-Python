"""Control-file parsing (replaces Read_Input_File in OGCM_DL.f90)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO


@dataclass
class Config:
    """Inputs supplied by the control file.

    Mirrors the variables read from stdin in `Read_Input_File()` of the Fortran
    program, but only the fields relevant to the ADCIRC paths (OutType >= 3).
    """

    ts: datetime
    te: datetime
    tmult: int
    bc_server: str
    out_type: int
    fort14: Path
    bc2d_name: Path

    def __post_init__(self) -> None:
        if self.out_type not in (3, 4, 5):
            raise ValueError(
                f"OutType={self.out_type} is not supported by this Python port. "
                "Use 3 (TS), 4 (TS+UV), or 5 (BCSL)."
            )
        if self.tmult < 1:
            raise ValueError(f"TMULT must be >= 1 (got {self.tmult})")
        if self.te < self.ts:
            raise ValueError("TE must be >= TS")


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")


def read_control_file(stream: IO[str] | str | Path) -> Config:
    """Parse the control file in the same line-by-line format as Fortran.

    Accepts either a path-like, an open text stream, or the file content as a
    string. The format matches `Read_Input_File()`:

    1. blank or comment
    2. TS  (YYYY-MM-DD HH:MM)
    3. TE  (YYYY-MM-DD HH:MM)
    4. TMULT (int)
    5. BCServer (3-char tag, ignored in the local-files port)
    6. OutType (must be 3, 4, or 5)
    7. fort.14 filename
    8. output BC2D NetCDF filename
    """

    if isinstance(stream, (str, Path)) and not isinstance(stream, io.IOBase):
        path = Path(stream)
        if path.exists():
            text = path.read_text()
        else:
            text = str(stream)
        stream = io.StringIO(text)

    lines = [ln.rstrip("\n") for ln in stream]  # type: ignore[arg-type]
    # Drop trailing empty lines so indexing is predictable
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 8:
        raise ValueError(
            f"Control file must have at least 8 lines (got {len(lines)})."
        )

    ts = _parse_dt(lines[1])
    te = _parse_dt(lines[2])
    tmult = int(lines[3].split()[0])
    bc_server = lines[4].strip()[:3]
    out_type = int(lines[5].split()[0])
    fort14 = Path(lines[6].strip())
    bc2d_name = Path(lines[7].strip())

    return Config(
        ts=ts,
        te=te,
        tmult=tmult,
        bc_server=bc_server,
        out_type=out_type,
        fort14=fort14,
        bc2d_name=bc2d_name,
    )
