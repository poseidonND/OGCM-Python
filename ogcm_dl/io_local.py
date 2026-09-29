"""Reads the local OGCM file index (replaces READOGCMDATA + GETOGCMFILE)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class OgcmIndex:
    """Index of available OGCM snapshots, parsed from `ogcm_data.txt`.

    Mirrors the OGCM_DATA derived type in the Fortran program. Each entry in
    `tsnap` is the time of the snapshot; `ts_files`, `uv_files`, `ssh_files`
    are the corresponding TS, UV, and SSH NetCDF filenames.
    """

    tsnap: list[datetime]
    ts_files: list[Path]
    uv_files: list[Path]
    ssh_files: list[Path]

    @property
    def n_snaps(self) -> int:
        return len(self.tsnap)

    def file_for(self, when: datetime, flag: int) -> Path:
        """Look up the file matching `when` for the given flag.

        `flag` follows the Fortran convention:
          - 1 -> TS file
          - 2 -> UV file
          - 3 -> SSH file (used by OutType=5 in the Fortran code via flag=1
                actually, but exposed here for completeness)
        """
        if when < self.tsnap[0] or when > self.tsnap[-1]:
            raise ValueError(
                f"Requested time {when:%Y-%m-%d %H:%M} is outside the OGCM "
                f"index range [{self.tsnap[0]:%Y-%m-%d %H:%M}, "
                f"{self.tsnap[-1]:%Y-%m-%d %H:%M}]."
            )
        # Find first kk such that tsnap[kk] <= when < tsnap[kk+1]; if when ==
        # the last snap, return the last entry. Matches GETOGCMFILE.
        kk = 0
        for i in range(self.n_snaps - 1):
            if self.tsnap[i] <= when < self.tsnap[i + 1]:
                kk = i
                break
        else:
            kk = self.n_snaps - 1
        if flag == 1:
            return self.ts_files[kk]
        elif flag == 2:
            return self.uv_files[kk]
        elif flag == 3:
            return self.ssh_files[kk]
        raise ValueError(f"Unknown OGCM file flag={flag}")


def read_ogcm_index(path: str | Path) -> OgcmIndex:
    """Read `ogcm_data.txt` (READOGCMDATA in the Fortran program).

    Format:

        <NSNAPS>
        <YYYY-MM-DD HH:MM>  <TS_FILE>  <UV_FILE>  <SSH_FILE>
        ...

    Whitespace is a delimiter. Quoted strings are accepted via shlex so that
    filenames with spaces are tolerated.
    """
    path = Path(path)
    raw = path.read_text().splitlines()
    raw = [ln for ln in raw if ln.strip()]
    if not raw:
        raise ValueError(f"{path} is empty")

    nsnaps = int(raw[0].split()[0])
    if len(raw) < 1 + nsnaps:
        raise ValueError(
            f"{path} declares {nsnaps} snaps but only {len(raw)-1} rows present."
        )

    tsnap: list[datetime] = []
    ts_files: list[Path] = []
    uv_files: list[Path] = []
    ssh_files: list[Path] = []

    for i in range(nsnaps):
        # Use shlex so quoted timestamps and paths are handled correctly.
        tokens = shlex.split(raw[1 + i])
        if len(tokens) < 5:
            raise ValueError(
                f"{path}:line {2+i}: expected '<date> <time> <ts> <uv> <ssh>' "
                f"(got {tokens!r})"
            )
        # Datetime is split by whitespace into "YYYY-MM-DD" and "HH:MM"
        date_str, time_str = tokens[0], tokens[1]
        when = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        tsnap.append(when)
        ts_files.append(Path(tokens[2]))
        uv_files.append(Path(tokens[3]))
        ssh_files.append(Path(tokens[4]))

    return OgcmIndex(
        tsnap=tsnap,
        ts_files=ts_files,
        uv_files=uv_files,
        ssh_files=ssh_files,
    )
