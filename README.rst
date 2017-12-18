=================
scikit-kinematics
=================

*scikit-kinematics* primarily contains functions for working with 3D
kinematics, e.g quaternions and rotation matrices.
This includes utilities to read in data from the following IMU-sensors:
    - XSens
    - xio
    - YEI

Compatible with 3 only (Python 2 is no longer tested).


Dependencies
------------
numpy, scipy, matplotlib, pandas, sympy, easygui

Homepage
--------
http://work.thaslwanter.at/skinematics/html/

Author:  Thomas Haslwanter
Date:    18-12-2017
Ver:     0.6.4
Licence: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
        Copyright (c) 2017, Thomas Haslwanter
        All rights reserved.

Installation
------------
You can install scikit-kinematics with

    pip install scikit-kinematics

and upgrade to a new version with

    pip install scikit-kinematics -U

IMUs
====

Analysis of signals from IMUs (intertial-measurement-units).
Read in data, calculate orientation (with one of the algorithms below)

- get_data ... This method must be taken from one of the existing sensors,
  or from your own sensor. Currenlty the following sensors types are available:
  * XSens
  * xio
  * yei
  * polulu
- calc_position

MARG Systems
------------
- imus.analytical ... Calculate orientation and position, from angular velocity and linear acceleration
- imus.kalman ... Calculate orientation from IMU-data using an Extended Kalman Filter.

- imus.IMU ... Class for working with data from IMUs
    - imus.IMU.calc_position ... calculate position
    - imus.IMU.setData ... set the properties of an IMU-object
- imus.Madgwick ... Class for calculating the 3D orientation with the Madgwick-algorithm
- imus.Mahony ... Class for calculating the 3D orientation with the Mahony-algorithm

Markers
=======

Analysis of signals from video-based marker-recordings of 3D movements

- markers.analyze_3Dmarkers ... Kinematic analysis of video-basedrecordings of 3D markers
- markers.find_trajectory ... Calculation of joint-movements from 3D marker positions

Quaternions
===========

Note that all these functions work with single quaternions and quaternion vectors,
as well as with arrays containing these.

Quaternion class
----------------

- quat.Quaternion ... class, including overloading for multiplication and
                    division (e.g. "quatCombined = quat1 * quat2"), import and export

Functions for working with quaternions
--------------------------------------

- quat.q_conj ... Conjugate quaternion 
- quat.q_inv ... Quaternion inversion
- quat.q_mult ... Quaternion multiplication
- quat.q_scalar ... Extract the scalar part from a quaternion
- quat.q_vector ... Extract the vector part from a quaternion
- quat.unit_q ... Extend a quaternion vector to a unit quaternion.

Conversion routines - quaternions
---------------------------------

- quat.calc_angvel ... Calculates the velocity in space from quaternions
- quat.calc_quat ... Calculate orientation from a starting orientation and angular velocity.
- quat.convert ... Convert quaternion to corresponding rotation matrix or Gibbs vector
- quat.deg2quat ... Convert number or axis angles to quaternion vectors
- quat.quat2seq ... Convert  quaternions to sequention rotations ("nautical" angles, etc)
- quat.scale2deg ... Convert quaternion to corresponding axis angle



Rotation Matrices
=================

Definition of rotation matrices
-------------------------------

- rotmat.R ... 3D rotation matrix for rotation about a coordinate axis

Conversion Routines - rotation matrices
---------------------------------------
- rotmat.convert ... Convert a rotation matrix to the corresponding quaternion
- rotmat.seq2quat ... Convert nautical angles etc. to quaternions
- rotmat.sequence ... Calculation of Euler, Fick, Helmholtz, ... angles

Symbolic matrices
-----------------

- rotmat.R_s() ... symbolix matrix for rotation about a coordinate axis

For example, you can e.g. generate a Fick-matrix, with

>>>    R_Fick = R_s(2, 'theta') * R_s(1, 'phi') * R_s(0, 'psi')
    
Vectors
=======

Routines for working with vectors
These routines can be used with vectors, as well as with matrices containing a vector in each row.

- vector.normalize ... Vector normalization
- vector.project ... Projection of one vector onto another one
- vector.GramSchmidt ... Gram-Schmidt orthogonalization of three points
- vector.q_shortest_rotation ... Quaternion indicating the shortest rotation from one vector into another.
- vector.rotate_vector ... Rotation of a vector
- vector.target2orient ... Convert target location into orientation angles

Interactive Data Analysis
=========================

- viewer.ts ... interactive viewer for time series data
