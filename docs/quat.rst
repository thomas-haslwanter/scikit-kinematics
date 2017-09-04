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

* :func:`quat.q_conj` ... Conjugate quaternion
* :func:`quat.q_inv` ... Quaternion inversion
* :func:`quat.q_mult` ... Quaternion multiplication
* :func:`quat.q_scalar` ... Extract the scalar part from a quaternion
* :func:`quat.q_vector` ... Extract the vector part from a quaternion
* :func:`quat.unit_q` ... Extend a quaternion vector to a unit quaternion.

Conversion routines
-------------------

* :func:`quat.calc_angvel` ... Calculates the velocity in space from quaternions
* :func:`quat.calc_quat` ... Calculate orientation from a starting orientation and angular velocity.
* :func:`quat.convert` ... Convert quaternion to corresponding rotation matrix or Gibbs vector
* :func:`quat.deg2quat` ... Convert number or axis angles to quaternion values
* :func:`quat.quat2deg` ... Convert quaternions to corresponding axis angle

.. toctree::
   :maxdepth: 2

Details
-------
.. automodule:: quat
    :members:
