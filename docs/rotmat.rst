.. _rotmat-label:

Rotation Matrices
=================

Definition of rotation matrices
-------------------------------

* :func:`rotmat.R1` ... 3D rotation matrix for rotation about the 1-axis
* :func:`rotmat.R2` ... 3D rotation matrix for rotation about the 2-axis
* :func:`rotmat.R3` ... 3D rotation matrix for rotation about the 3-axis

Conversion Routines
-------------------
* :func:`rotmat.rotmat2Euler` ... Calculation of Euler angles
* :func:`rotmat.rotmat2Fick` ... Calculation of Fick angles
* :func:`rotmat.rotmat2Helmholtz` ... Calculation of Helmholtz angles

Symbolic matrices
-----------------

* :func:`rotmat.R1_s()` ... symbolix matrix for rotation about the 1-axis
* :func:`rotmat.R2_s()` ... symbolix matrix for rotation about the 2-axis
* :func:`rotmat.R3_s()` ... symbolix matrix for rotation about the 3-axis

For example, you can e.g. generate a Fick-matrix, with

    R_Fick = R3_s() * R2_s() * R1_s()

.. toctree::
   :maxdepth: 2

Details
-------

.. automodule:: rotmat
    :members:
