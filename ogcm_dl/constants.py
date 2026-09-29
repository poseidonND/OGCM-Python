"""Physical and numerical constants matching the Fortran program."""

import math

import numpy as np

RHO_WAT0: float = 1.0e3  # reference density [kg m^-3]
G: float = 9.80665  # gravity [m s^-2]
R_EARTH: float = 6378206.4  # Earth radius used by the Fortran program [m]

DEG2RAD: float = math.pi / 180.0
RAD2DEG: float = 180.0 / math.pi

# Upper-bound for latitude in haversine (matches Fortran)
LAT_UL: float = 89.0

# Upper/lower bound for baroclinic pressure gradients [m s^-2]
BPG_LIM: float = 0.1

# Fill values used by the Fortran program for output NetCDF
SIG_T0: float = -999.0  # sentinel for surface sigma-t
DFV: float = 1.0  # default fill for MLD ratio (matches Fortran DFV)

# Sentinels used internally for invalid 3D values (Fortran FV/FVP)
FV: float = -3.0e4
FVP: float = -3.0e4 + 1.0e-3

# Default output dtype (Fortran sz=8 / nf90_double)
NETCDF_DTYPE = np.float64

# Compression level for NetCDF4 (matches Fortran dfl=2)
DEFLATE_LEVEL: int = 2

# Default cadence of the OGCM in hours (matches Fortran BC3D_DT=3)
BC3D_DT_HOURS: int = 3

# Mercator projection reference (Fortran SLAM0=0, SFEA0=0)
SLAM0: float = 0.0
SFEA0: float = 0.0
