'''
"scikit-kinematics" primarily contains functions for working with 3D kinematics. (i.e.
quaternions and rotation matrices).

Compatible with Python 2 and 3.

Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, easygui, libdeprecation

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Copyright (c) 2016 Thomas Haslwanter <thomas.haslwanter@fh-linz.at>

'''

import importlib

__author__ = "Thomas Haslwanter <thomas.haslwanter@fh-linz.at"
__license__ = "BSD 2-Clause License"
__version__ = "0.4.2"

__all__ = ['imus', 'markers', 'misc', 'quat', 'rotmat', 'vector', 'view', 'sensors']
#__all__ = []

for _m in __all__:
    importlib.import_module('.'+_m, package='skinematics')
