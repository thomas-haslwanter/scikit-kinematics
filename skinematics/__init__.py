"""
"scikit-kinematics" primarily contains functions for working with 3D kinematics. (i.e.
quaternions and rotation matrices).

Compatible with Python >= 3.12.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, deprecated
Optional (for view.Orientation_OGL): pygame-ce, PyOpenGL

Homepage
--------
https://work.thaslwanter.at/skinematics/html/

Copyright (c) 2026 Thomas Haslwanter <office@thaslwanter.at>

"""

import importlib

__author__ = "Thomas Haslwanter <office@thaslwanter.at>"
__license__ = "BSD 3-Clause License"
__version__ = "0.12.1"


required_imports = ["markers", "quat", "rotmat", "vector", "sensors"]
optional_imports = ["imus", "misc"]

__all__ = []
for _m in required_imports:
    importlib.import_module("." + _m, package="skinematics")
    __all__.append(_m)

for _m in optional_imports:
    try:
        importlib.import_module("." + _m, package="skinematics")
        __all__.append(_m)
    except ModuleNotFoundError:
        print(f"Failed to import optional module {_m}. Install optional dependencies")
