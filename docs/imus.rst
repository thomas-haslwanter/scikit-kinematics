.. _imus-label:

IMUs
====

These routines facilitate the calculation of 3d movement kinematics for
recordings from inertial measurement units (IMUs).
Currently data from 4 systems are supported:

* XSens
* xio
* YEI
* polulu

Functions
---------

Data-Handling
^^^^^^^^^^^^^
* :func:`imus.import_data` ... Read in rate and 3D parameters from different IMUs.
* :func:`sensors.xio.get_data` ... Read in rate and 3D parameters from *xio* sensors.
* :func:`sensors.xsens.get_data` ... Read in rate and 3D parameters from *XSens* sensors.
* :func:`sensors.yei.get_data` ... Read in rate and 3D parameters from *YEI* sensors.
* :func:`sensors.polulu.get_data` ... Read in rate and 3D parameters from *polulu* sensors.

Data-Analysis
^^^^^^^^^^^^^
* :func:`imus.analytical` ... Calculate orientation and position analytically from angular velocity and linear acceleration 
* :func:`imus.kalman` ... Calculate orientation from IMU-data using an Extended Kalman Filter

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
    imus.Mahony
    imus.Madgwick

Details
-------
.. automodule:: imus
    :members:

.. automodule:: sensors.xio
    :members:

.. automodule:: sensors.xsens
    :members:

.. automodule:: sensors.yei
    :members:

.. automodule:: sensors.polulu
    :members:

