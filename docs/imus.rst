.. _imus-label:

IMUs
====

These routines facilitate the calculation of 3d movement kinematics for
recordings from inertial measurement units (IMUs).

They are implemented in an object oriented way. (Don't worry if you have not
used objects before, it won't be a problem here!) All sensor implementations
are based on the abstract base class "IMU_Base". For each sensor, the
corresponding method "get_data" has to be implemented, by sub-classing
IMU_Base. Currently 5 sensor types are supported:

* XSens
* xio
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

Sub-classing IMU-Base for your own sensor type
----------------------------------------------
If you have your own data format, you have to implement the corresponding
"get_data" method. You can base it on:

* "xsens.py" ... if all your data are in one file
* "polulu.py" ... if you have to manually enter data not stored by your program
* "xio.py" ... if your data are stored in multiple files

Class
-----
.. autosummary::
    imus.IMU_Base

.. automodule:: imus
    :members:


Methods
^^^^^^^
.. autosummary::
    imus.IMU_Base.calc_position
    imus.IMU_Base.get_data

.. toctree::
   :maxdepth: 2

Classes for Sensor-Integration
------------------------------
.. autosummary::
    imus.Mahony
    imus.Madgwick

Functions
---------

Data-Analysis
^^^^^^^^^^^^^
* :func:`imus.analytical` ... Calculate orientation and position analytically from angular velocity and linear acceleration 
* :func:`imus.kalman` ... Calculate orientation from IMU-data using an Extended Kalman Filter

.. toctree::
   :maxdepth: 2

Existing Sensor Implementations
-------------------------------

.. automodule:: sensors.xio
    :members:

.. automodule:: sensors.xsens
    :members:

.. automodule:: sensors.yei
    :members:

.. automodule:: sensors.polulu
    :members:

