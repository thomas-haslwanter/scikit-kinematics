'''
"scikit-kinematics" primarily contains functions for working with 3D kinematics. (i.e.
quaternions and rotation matrices).

Compatible with Python 2 and 3.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, easygui

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2016 Thomas Haslwanter <thomas.haslwanter@fh-linz.at>

'''

import importlib

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at"
__license__ = "BSD 2-Clause License"
__version__ = "0.1.2"

__all__ = ['imus', 'markers', 'quat', 'rotmat', 'vector', 'viewer']

for _m in __all__:
    importlib.import_module('.'+_m, package='skinematics')
