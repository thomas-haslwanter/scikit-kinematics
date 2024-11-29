.. _inertial-label:

IMUs
====

These routines facilitate the calculation of 3d movement kinematics for
recordings from inertial measurement units (IMUs).

They are implemented in an object oriented way. (Don't worry if you have not
used objects before, it won't be a problem here!) All sensor implementations
are based on the abstract base class "IMU_Base". For each sensor, the
corresponding method "get_data" has to be implemented, by sub-classing
IMU_Base. Currently the following sensor types are supported:

* XSens
* xio (XIO, NGIMU, and XIO3)
* YEI
* polulu
* manual

The last one is not a "real" sensor, but allows the creation of an
IMU-object with your own IMU-data, without defining a new class.
To create a sensor object, choose one of the existing sensor classes, as
demonstrated in the example below. You have to provide at least the
file-name of the file containing the sensor data. Optionally, you can also
provide:

* R_init ... initial orientation [default = np.eye(3)]
* pos_init ... initial position [default = np.ones(3)]
* q_type ... method to calculate orientation. The options are:
  
    - "analytical" [default] ... analytical solution, using only acc and omega
    - "kalman" ... quaternion Kalman filter, using acc, omega, and mag
    - "madgwick" ... Madgwick algorithm, using acc, omega, and mag
    - "mahony" ... Mahony algorithm, using, acc and omega, and mag
    - "None" ... If you want to only read in the sensor data

Data are read in, and by default the orientation is automatically calculated
based on the parameter "q_type" and using the function _calc_orientation.

Note: support vor "Vicon" has dropped, because the Vicon btk is no longer supported.

Base-Class & Methods
--------------------

.. autosummary::
    imus.IMU_Base
    imus.IMU_Base.calc_position
    imus.IMU_Base.get_data
    imus.IMU_Base.set_qtype

Classes and Functions for Sensor-Integration
--------------------------------------------

* :func:`imus.analytical` ... Calculate orientation and position analytically from angular velocity and linear acceleration 
* :func:`imus.kalman` ... Calculate orientation from IMU-data using an Extended Kalman Filter

.. autosummary::
    imus.Mahony
    imus.Madgwick

Details
-------

.. automodule:: imus
    :members:


Sub-classing IMU-Base for your own sensor type
----------------------------------------------
If you have your own data format, you have to implement the corresponding
"get_data" method. You can base it on:

* "xsens.py" ... if all your data are in one file
* "polulu.py" ... if you have to manually enter data not stored by your program
* "xio.py" ... if your data are stored in multiple files

Existing Sensor Implementations
-------------------------------

XIO
^^^
.. automodule:: sensors.xio
    :members:

x-io NGIMU
----------
.. automodule:: sensors.xio_ngimu
    :members:

XSens
^^^^^
.. automodule:: sensors.xsens
    :members:

YEI
^^^
.. automodule:: sensors.yei
    :members:

Polulu
^^^^^^
.. automodule:: sensors.polulu
    :members:


