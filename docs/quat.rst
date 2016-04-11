.. _quat-label:

Quaternions
===========

Note that all these functions work with single quaternions and quaternion vectors,
as well as with arrays containing these.

Quaternion class
----------------

* :class:`quat.Quaternion` ... Quaternion class

Functions for working with quaternions
--------------------------------------

* :func:`quat.quatconj` ... Conjugate quaternion
* :func:`quat.quatinv` ... Quaternion inversion
* :func:`quat.quatmult` ... Quaternion multiplication

Conversion routines
-------------------

* :func:`quat.deg2quat` ... Convert number or axis angles to quaternion vectors
* :func:`quat.quat2deg` ... Convert quaternion to corresponding axis angle
* :func:`quat.quat2rotmat` ... Convert quaternion to corresponding rotation matrix
* :func:`quat.quat2vect` ... Extract the vector part from a quaternion
* :func:`quat.quat2vel` ... Calculates the velocity in space from quaternions
* :func:`quat.rotmat2quat` ... Convert a rotation matrix to the corresponding quaternion
* :func:`quat.vect2quat` ... Extend a quaternion vector to a unit quaternion.
* :func:`quat.vel2quat` ... Calculate orientation from a starting orientation and angular velocity.

.. toctree::
   :maxdepth: 2

Details
-------
.. automodule:: quat
    :members:
