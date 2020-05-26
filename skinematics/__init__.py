'''
"scikit-kinematics" primarily contains functions for working with 3D kinematics. (i.e.
quaternions and rotation matrices).

Compatible Python 3.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, pygame, pyOpenGL

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2018 Thomas Haslwanter <thomas.haslwanter@fh-ooe.at>

'''

import importlib

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at"
__license__ = "BSD 2-Clause License"
__version__ = "0.8.3"

__all__ = ['imus', 'markers', 'misc', 'quat', 'rotmat', 'vector', 'view', 'sensors']

for _m in __all__:
    try:
        importlib.import_module('.'+_m, package='skinematics')
    except:
        print(f"Failed to automatically import {_m}.")
