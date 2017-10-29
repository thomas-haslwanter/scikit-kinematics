.. _rotmat-label:

Rotation Matrices
=================

Definition of rotation matrices
-------------------------------

* :func:`rotmat.R` ... 3D rotation matrix for rotation about a coordinate axis

Conversion Routines
-------------------
* :func:`rotmat.convert` ... Convert a rotation matrix to the corresponding quaternion
* :func:`rotmat.sequence` ... Calculation of Euler, Fick/aeronautic, Helmholtz angles
* :func:`rotmat.seq2quat` ... Calculation of quaternions from Euler, Fick/aeronautic, Helmholtz angles

Symbolic matrices
-----------------

* :func:`rotmat.R_s()` ... symbolix matrix for rotation about a coordinate axis

For example, you can e.g. generate a Fick-matrix, with

    R_Fick = R_s(2, 'theta') * R_s(1, 'phi') * R_s(0, 'psi')

.. toctree::
   :maxdepth: 2

Details
-------

.. automodule:: rotmat
    :members:
