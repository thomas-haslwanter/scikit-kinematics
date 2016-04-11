.. _imus-label:

IMUs
====

These routines facilitate the calculation of 3d movement kinematics for
recordings from inertial measurement units (IMUs).

Functions
---------

General
^^^^^^^
* :func:`imus.getXSensData` ... Read in Rate and stored 3D parameters from XSens IMUs

IMUs
^^^^
* :func:`imus.calc_QPos` ... Calculate orientation and position, from angular velocity and linear acceleration 
* :func:`imus.kalman_quat` ... Calculate orientation from IMU-data using an Extended Kalman Filter

.. toctree::
   :maxdepth: 2

Class
-----
.. autosummary::
    imus.IMU


Methods
^^^^^^^
.. autosummary::

    imus.IMU.calc_orientation
    imus.IMU.calc_position
    imus.IMU.setData

.. toctree::
   :maxdepth: 2

Class
-----
.. autosummary::
    imus.MadgwickAHRS

Class
-----
.. autosummary::
    imus.MahonyAHRS

Details
-------
.. automodule:: imus
    :members:
