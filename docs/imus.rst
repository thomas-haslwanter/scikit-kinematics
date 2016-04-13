.. _imus-label:

IMUs
====

These routines facilitate the calculation of 3d movement kinematics for
recordings from inertial measurement units (IMUs).
Currently data from two systems are supported:

* XSens
* xio

Functions
---------

Data-Handling
^^^^^^^^^^^^^
* :func:`imus.import_data` ... Read in rate and 3D parameters from different IMUs.

Data-Analysis
^^^^^^^^^^^^^
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

Classes for Sensor-Integration
------------------------------
.. autosummary::
    imus.MahonyAHRS
    imus.MadgwickAHRS

Details
-------
.. automodule:: imus
    :members:
