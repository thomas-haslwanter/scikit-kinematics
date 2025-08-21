"""
"scikit-kinematics" primarily contains functions for working with 3D kinematics. (i.e.
quaternions and rotation matrices).

Compatible Python 3.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, pygame, pyOpenGL

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2025 Thomas Haslwanter <thomas.haslwanter@fh-ooe.at>

"""

import importlib

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at"
__license__ = "BSD 2-Clause License"
__version__ = "0.10.4"


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
