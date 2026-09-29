"""Python port of OGCM_DL.f90 (ADCIRC-grid path only).

See README.md for usage.

Heavy dependencies (`numpy`, `netCDF4`, `gsw`) are imported lazily so that
just `import ogcm_dl` (or `from ogcm_dl.config import ...`) works without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config
    from .runner import run

__all__ = ["Config", "read_control_file", "run"]


def __getattr__(name: str):  # PEP 562: lazy attribute access
    if name in ("Config", "read_control_file"):
        from . import config as _config
        return getattr(_config, name)
    if name == "run":
        from . import runner as _runner
        return _runner.run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
