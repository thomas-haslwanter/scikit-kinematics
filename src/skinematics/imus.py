"""
This file contains the abstract base class "IMU_Base" for analyzing movements
recordings with inertial measurement units (IMUs), as well as functions and
classes for the evaluation of IMU-data..

The advantage of using an "abstract base class" is that it allows to write
code that is independent of the IMU-sensor. All IMUs provide acceleration
and angular velocities, and most of them also the direction of the local
magnetic field. The specifics of each sensor are hidden in the sensor-object
(specifically, in the "get\_data" method which has to be implemented once
for each sensor). Initialization of a sensor object includes a number of
activities:

        - Reading in the data.
        - Making acceleration, angular\_velocity etc. accessible in a sensor-
          independent way
        - Calculating duration, totalSamples, etc.
        - Calculating orientation (expressed as "quat"), with the method
          specified in "q\_type"

"""

#Author: Thomas Haslwanter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy as sp
from scipy import constants     # for "g"
from scipy.integrate import cumtrapz
import re

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys

file_dir = os.path.dirname(__file__)
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

import quat, vector, misc, rotmat

# For deprecation warnings
# import deprecation
import warnings

# For the definition of the abstract base class IMU_Base
import abc

class IMU_Base(metaclass=abc.ABCMeta):
    '''
    Abstract BaseClass for working with working with inertial measurement units (IMUs)
    A concrete class must be instantiated, which implements "get_data".
    (See example below.)

    Attributes:
        acc (Nx3 array) : 3D linear acceleration [m/s**2]
        dataType (string) : Type of data (commonly float)
        duration (float) : Duration of recording [sec]
        mag (Nx3 array) : 3D orientation of local magnectic field
        omega (Nx3 array) : 3D angular velocity [rad/s]
        pos (Nx3 array) : 3D position
        pos_init (3-vector) : Initial position. default is np.ones(3)
        quat (Nx4 array) : 3D orientation
        q_type (string) : Method of calculation for orientation quaternion
        rate (float) : Sampling rate [Hz]
        R_init (3x3 array) : Rotation matrix defining the initial orientation.
                             Default is np.eye(3)
        source (str) : Name of data-file
        totalSamples (int) : Number of samples
        vel (Nx3 array) : 3D velocity

    Parameters
    ----------
    inFile : string
        path- and file-name of data file / input source
    inData : dictionary
        The following fields are required:

        acc : (N x 3) array
             Linear acceleration [m/s^2], in the x-, y-, and z-direction
        omega : (N x 3) array
             Angular velocity [rad/s], about the x-, y-, and z-axis
        [mag] : (N x 3) array (optional)
             Local magnetic field, in the x-, y-, and z-direction
        rate: float
            sample rate [Hz]


    Examples
    --------
    >>> # Set the in-file, initial sensor orientation 
    >>> in_file = r'tests/data/data_xsens.txt'
    >>> initial_orientation = np.array([[1,0,0],
    >>>                                 [0,0,-1],
    >>>                                 [0,1,0]])
    >>>  
    >>> # Choose a sensor 
    >>> from skinematics.sensors.xsens import XSens
    >>> from skinematics.sensors.manual import MyOwnSensor
    >>>
    >>> # Only read in the data
    >>> data = XSens(in_file, q_type=None)
    >>>
    >>> # Read in and evaluate the data
    >>> sensor = XSens(in_file=in_file, R_init=initial_orientation)
    >>>  
    >>> # By default, the orientation quaternion gets automatically calculated,
    >>> #    using the option "analytical"
    >>> q_analytical = sensor.quat
    >>>  
    >>> # Automatic re-calculation of orientation if "q_type" is changed
    >>> sensor.set_qtype('madgwick')
    >>> q_Madgwick = sensor.quat
    >>>  
    >>> sensor.set_qtype('kalman')
    >>> q_Kalman = sensor.quat
    >>>
    >>> # Demonstrate how to fill up a sensor manually
    >>> in_data = {'rate':sensor.rate,
    >>>         'acc': sensor.acc,
    >>>         'omega':sensor.omega,
    >>>         'mag':sensor.mag}
    >>> my_sensor = MyOwnSensor(in_file='My own 123 sensor.', in_data=in_data)

    '''

    @abc.abstractmethod
    def get_data(self, in_file=None, in_data=None):
        """Retrieve "rate", "acc", "omega", "mag" from the input source
        and set the corresponding values of "self".
        With some sensors, "rate" has to be provided, and is taken from "in_data".
        """

    def __init__(self, in_file = None,
                 q_type='analytical', R_init = np.eye(3),
                 calculate_position=True, pos_init = np.zeros(3),
                 in_data = None ):
        """Initialize an IMU-object.
        Note that this includes a number of activities:
        - Read in the data (which have to be either in "in_file" or in "in_data")
        - Make acceleration, angular_velocity etc. accessible, in a
          sensor-independent way
        - Calculates duration, totalSamples, etc
        - If q_type==None, data are only read in; otherwise, 3-D orientation is
          calculated with the method specified in "q_type", and stored in the
          property "quat".
        - If position==True, the method "calc_position" is automatically called,
          and the 3D position stored in the propery "pos". (Note that
          if q_type==None, then "position" is set to "False".)

        in_file : string
                Location of infile / input
        q_type : string
                Determines how the orientation gets calculated:
                - 'analytical' .... quaternion integration of angular velocity [default]
                - 'kalman' ..... quaternion Kalman filter
                - 'madgwick' ... gradient descent method, efficient
                - 'mahony' ....  formula from Mahony, as implemented by Madgwick
                - 'None' ... data are only read in, no orientation calculated
        R_init : 3x3 array
                approximate alignment of sensor-CS with space-fixed CS
                currently only used in "analytical"
        calculate_position : Boolean
                If "True", position is calculated, and stored in property "pos".
        pos_init : (,3) vector
                initial position
                currently only used in "analytical"
        in_data : dictionary
                If the data are provided directly, not from a file
                Also used to provide "rate" for "polulu" sensors.
        """

        if in_data is None and in_file is None:
            raise ValueError('Either in_data or in_file must be provided.')

        elif in_file is None:
            # Get the data from "in_data"
            # In that case, "in_data"
            #   - must contain the fields ['acc', 'omega']
            #   - can contain the fields ['rate', 'mag']
            # self.source is set to "None"
            self._set_data(in_data)

        else: 
            # Set rate, acc, omega, mag
            # Note: this is implemented in the concrete class, implenented in 
            # the corresponding module in "sensors"
            self.source = in_file
            self.get_data(in_file, in_data)

        # Set information not determined by the IMU-data
        self.R_init = R_init
        self.pos_init = pos_init

        # Set the analysis method, and consolidate the object (see below)
        # This also means calculating the orientation quaternion!!
        self.set_qtype(q_type)
        
        if q_type != None:
            if calculate_position:
                self.calc_position()

    def set_qtype(self, type_value):                
        """Sets q_type, and automatically performs the relevant calculations.
        q_type determines how the orientation is calculated.
        If "q_type" is "None", no orientation gets calculated; otherwise,
        the orientation calculation is performed with 
        "_calc_orientation", using the option "q_type".

        It has to be one of the following values:

        * analytical
        * kalman
        * madgwick
        * mahony
        * None

        """
        
        allowed_values = ['analytical',
                          'kalman',
                          'madgwick',
                          'mahony',
                          None]
        
        if type_value in allowed_values:
            self.q_type = type_value
            if type_value == None:
                self.quat = None
            else:
                self._calc_orientation()
        else:
            raise ValueError('q_type must be one of the following: ' \
                             '{0}, not {1}'.format(allowed_values, value))
        

    def _set_data(self, data):
        # Set the properties of an IMU-object directly

        if 'rate' not in data.keys():
            print('Set the "rate" to the default value (100 Hz).')
            data['rate'] = 100.0

        self.rate = data['rate']
        self.acc= data['acc']
        self.omega = data['omega']
        if 'mag' in data.keys():
            self.mag = data['mag']
        self.source = None
        self._set_info()


    def _calc_orientation(self):
        '''
        Calculate the current orientation

        Parameters
        ----------
        type : string
                - 'analytical' .... quaternion integration of angular velocity
                - 'kalman' ..... quaternion Kalman filter
                - 'madgwick' ... gradient descent method, efficient
                - 'mahony' ....  formula from Mahony, as implemented by Madgwick

        '''

        initialPosition = np.r_[0,0,0]

        method = self.q_type
        if method == 'analytical':
            (quaternion, position, velocity) = analytical(self.R_init,
                                                          self.omega,
                                                          initialPosition,
                                                          self.acc,
                                                          self.rate) 

        elif method == 'kalman':
            self._checkRequirements()
            quaternion = kalman(self.rate, self.acc, np.deg2rad(self.omega),
                                self.mag)

        elif method == 'madgwick':
            self._checkRequirements()

            # Initialize object
            AHRS = Madgwick(rate=self.rate, Beta=0.5)    # previously: Beta=1.5
            quaternion = np.zeros((self.totalSamples, 4))

            # The "Update"-function uses angular velocity in radian/s, and only
            # the directions of acceleration and magnetic field
            Gyr = self.omega
            Acc = vector.normalize(self.acc)
            Mag = vector.normalize(self.mag)

            #for t in range(len(time)):
            for t in misc.progressbar(range(self.totalSamples),
                                      'Calculating the Quaternions ', 25) :
                AHRS.Update(Gyr[t], Acc[t], Mag[t])
                quaternion[t] = AHRS.Quaternion

        elif method == 'mahony':
            self._checkRequirements()

            # Initialize object
            AHRS = Mahony(rate=float(self.rate), Kp=0.4)  # previously: Kp=0.5
            quaternion = np.zeros((self.totalSamples, 4))

            # The "Update"-function uses angular velocity in radian/s, and only
            # the directions of acceleration and magnetic field
            Gyr = self.omega
            Acc = vector.normalize(self.acc)
            Mag = vector.normalize(self.mag)

            #for t in range(len(time)):
            for t in misc.progressbar(range(self.totalSamples),
                                      'Calculating the Quaternions ', 25) :
                AHRS.Update(Gyr[t], Acc[t], Mag[t])
                quaternion[t] = AHRS.Quaternion

        else:
            print('Unknown orientation type: {0}'.format(method))
            return

        self.quat = quaternion


    def calc_position(self):
        '''Calculate the position, assuming that the orientation is already known.'''

        initialPosition = self.pos_init
        # Acceleration, velocity, and position ----------------------------
        # From q and the measured acceleration, get the \frac{d^2x}{dt^2}
        g = constants.g
        g_v = np.r_[0, 0, g] 
        accReSensor = self.acc - vector.rotate_vector(g_v, quat.q_inv(self.quat))
        accReSpace = vector.rotate_vector(accReSensor, self.quat)

        # Position and Velocity through integration, assuming 0-velocity at t=0
        vel = np.nan*np.ones_like(accReSpace)
        pos = np.nan*np.ones_like(accReSpace)

        for ii in range(accReSpace.shape[1]):
            vel[:,ii] = cumtrapz(accReSpace[:,ii],
                                 dx=1./float(self.rate), initial=0)
            pos[:,ii] = cumtrapz(vel[:,ii],
                        dx=1./float(self.rate), initial=initialPosition[ii])

        self.vel = vel
        self.pos = pos

    def _checkRequirements(self):
        '''Check if all the necessary variables are available.'''
        required = [ 'rate', 'acc', 'omega', 'mag' ]

        for field in required:
            if field not in vars(self):
                print('Cannot find {0} in calc_orientation!'.format(field))

    def _set_info(self):
        '''Complete the information properties of that IMU'''

        self.totalSamples = len(self.omega)
        self.duration = float(self.totalSamples)/self.rate # [sec]
        self.dataType = str(self.omega.dtype)


def analytical(R_initialOrientation=np.eye(3),
               omega=np.zeros((5,3)),
               initialPosition=np.zeros(3),
               accMeasured=np.column_stack((np.zeros((5,2)), 9.81*np.ones(5))),
               rate=100):
    ''' Reconstruct position and orientation with an analytical solution,
    from angular velocity and linear acceleration.
    Assumes a start in a stationary position. No compensation for drift.

    Parameters
    ----------
    R_initialOrientation: ndarray(3,3)
        Rotation matrix describing the initial orientation of the sensor,
        except a mis-orienation with respect to gravity
    omega : ndarray(N,3)
        Angular velocity, in [rad/s]
    initialPosition : ndarray(3,)
        initial Position, in [m]
    accMeasured : ndarray(N,3)
        Linear acceleration, in [m/s^2]
    rate : float
        sampling rate, in [Hz]

    Returns
    -------
    q : ndarray(N,3)
        Orientation, expressed as a quaternion vector
    pos : ndarray(N,3)
        Position in space [m]
    vel : ndarray(N,3)
        Velocity in space [m/s]

    Example
    -------
     
    >>> q1, pos1 = analytical(R_initialOrientation, omega, initialPosition, acc, rate)

    '''

    if omega.ndim == 1:
        raise ValueError('The input to "analytical" requires matrix inputs.')
        
    # Transform recordings to angVel/acceleration in space --------------

    # Orientation of \vec{g} with the sensor in the "R_initialOrientation"
    g = constants.g
    g0 = np.linalg.inv(R_initialOrientation).dot(np.r_[0,0,g])

    # for the remaining deviation, assume the shortest rotation to there
    q0 = vector.q_shortest_rotation(accMeasured[0], g0)    
    
    q_initial = rotmat.convert(R_initialOrientation, to='quat')
    
    # combine the two, to form a reference orientation. Note that the sequence
    # is very important!
    q_ref = quat.q_mult(q_initial, q0)
    
    # Calculate orientation q by "integrating" omega -----------------
    q = quat.calc_quat(omega, q_ref, rate, 'bf')

    # Acceleration, velocity, and position ----------------------------
    # From q and the measured acceleration, get the \frac{d^2x}{dt^2}
    g_v = np.r_[0, 0, g] 
    accReSensor = accMeasured - vector.rotate_vector(g_v, quat.q_inv(q))
    accReSpace = vector.rotate_vector(accReSensor, q)

    # Make the first position the reference position
    q = quat.q_mult(q, quat.q_inv(q[0]))

    # compensate for drift
    #drift = np.mean(accReSpace, 0)
    #accReSpace -= drift*0.7

    # Position and Velocity through integration, assuming 0-velocity at t=0
    vel = np.nan*np.ones_like(accReSpace)
    pos = np.nan*np.ones_like(accReSpace)

    for ii in range(accReSpace.shape[1]):
        vel[:,ii] = cumtrapz(accReSpace[:,ii], dx=1./rate, initial=0)
        pos[:,ii] = cumtrapz(vel[:,ii], dx=1./rate, initial=initialPosition[ii])

    return (q, pos, vel)

def kalman(rate, acc, omega, mag,
           D = [0.4, 0.4, 0.4],          
           tau = [0.5, 0.5, 0.5],
           Q_k = None,
           R_k = None):
    '''
    Calclulate the orientation from IMU magnetometer data.

    Parameters
    ----------
    rate : float
    	   sample rate [Hz]	
    acc : (N,3) ndarray
    	  linear acceleration [m/sec^2]
    omega : (N,3) ndarray
    	  angular velocity [rad/sec]
    mag : (N,3) ndarray
    	  magnetic field orientation
    D : (,3) ndarray
          noise variance, for x/y/z [rad^2/sec^2]
          parameter for tuning the filter; defaults from Yun et al.
          can also be entered as list
    tau : (,3) ndarray
          time constant for the process model, for x/y/z [sec]
          parameter for tuning the filter; defaults from Yun et al.
          can also be entered as list
    Q_k : None, or (7,7) ndarray
          covariance matrix of process noises
          parameter for tuning the filter
          If set to "None", the defaults from Yun et al. are taken!
    R_k : None, or (7,7) ndarray
          covariance matrix of measurement noises
          parameter for tuning the filter; defaults from Yun et al.
          If set to "None", the defaults from Yun et al. are taken!
          

    Returns
    -------
    qOut : (N,4) ndarray
    	   unit quaternion, describing the orientation relativ to the coordinate
           system spanned by the local magnetic field, and gravity

    Notes
    -----
    Based on "Design, Implementation, and Experimental Results of a Quaternion-
       Based Kalman Filter for Human Body Motion Tracking" Yun, X. and Bachman,
       E.R., IEEE TRANSACTIONS ON ROBOTICS, VOL. 22, 1216-1227 (2006)

    '''

    numData = len(acc)

    # Set parameters for Kalman Filter
    tstep = 1./rate
    
    # check input
    assert len(tau) == 3
    tau = np.array(tau)

    # Initializations 
    x_k = np.zeros(7)	# state vector
    z_k = np.zeros(7)   # measurement vector
    z_k_pre = np.zeros(7)
    P_k = np.eye(7)		 # error covariance matrix P_k

    Phi_k = np.eye(7)    # discrete state transition matrix Phi_k
    for ii in range(3):
        Phi_k[ii,ii] = np.exp(-tstep/tau[ii])

    H_k = np.eye(7)		# Identity matrix

    D = np.r_[0.4, 0.4, 0.4]		# [rad^2/sec^2]; from Yun, 2006
    
    if Q_k is None:
        # Set the default input, from Yun et al.
        Q_k = np.zeros((7,7)) 	# process noise matrix Q_k
        for ii in range(3):
            Q_k[ii,ii] =  D[ii]/(2*tau[ii])  * ( 1-np.exp(-2*tstep/tau[ii]) )
    else:
        # Check the shape of the input
        assert Q_k.shape == (7,7)

    # Evaluate measurement noise covariance matrix R_k
    if R_k is None:
        # Set the default input, from Yun et al.
        r_angvel = 0.01;      # [rad**2/sec**2]; from Yun, 2006
        r_quats = 0.0001;     # from Yun, 2006
        
        r_ii = np.zeros(7)
        for ii in range(3):
            r_ii[ii] = r_angvel
        for ii in range(4):
            r_ii[ii+3] = r_quats
            
        R_k = np.diag(r_ii)    
    else:
        # Check the shape of the input
        assert R_k.shape == (7,7)

    # Calculation of orientation for every time step
    qOut = np.zeros( (numData,4) )

    for ii in range(numData):
        accelVec  = acc[ii,:]
        magVec    = mag[ii,:]
        angvelVec = omega[ii,:]
        z_k_pre = z_k.copy()  # watch out: by default, Python passes the reference!!

        # Evaluate quaternion based on acceleration and magnetic field data 
        accelVec_n = vector.normalize(accelVec)
        magVec_hor = magVec - accelVec_n * (accelVec_n @ magVec)
        magVec_n   = vector.normalize(magVec_hor)
        basisVectors = np.column_stack( [magVec_n,
                                np.cross(accelVec_n, magVec_n), accelVec_n] )
        quatRef = quat.q_inv(rotmat.convert(basisVectors, to='quat')).ravel()

        # Calculate Kalman Gain
        # K_k = P_k * H_k.T * inv(H_k*P_k*H_k.T + R_k)
        K_k = P_k @ np.linalg.inv(P_k + R_k)

        # Update measurement vector z_k
        z_k[:3] = angvelVec
        z_k[3:] = quatRef

        # Update state vector x_k
        x_k += np.array( K_k@(z_k-z_k_pre) ).ravel()

        # Evaluate discrete state transition matrix Phi_k
        Delta = np.zeros((7,7))
        Delta[3,:] = np.r_[-x_k[4], -x_k[5], -x_k[6],      0, -x_k[0], -x_k[1], -x_k[2]]
        Delta[4,:] = np.r_[ x_k[3], -x_k[6],  x_k[5], x_k[0],       0,  x_k[2], -x_k[1]]
        Delta[5,:] = np.r_[ x_k[6],  x_k[3], -x_k[4], x_k[1], -x_k[2],       0,  x_k[0]]
        Delta[6,:] = np.r_[-x_k[5],  x_k[4],  x_k[3], x_k[2],  x_k[1], -x_k[0],       0]
        
        Delta *= tstep/2
        Phi_k += Delta

        # Update error covariance matrix
        P_k = (np.eye(7) - K_k) @ P_k

        # Projection of state
        # 1) quaternions
        x_k[3:] += tstep * 0.5 * quat.q_mult(x_k[3:], np.r_[0, x_k[:3]]).ravel()
        x_k[3:] = vector.normalize( x_k[3:] )
        # 2) angular velocities
        x_k[:3] -= tstep * tau * x_k[:3]

        qOut[ii,:] = x_k[3:]

        # Projection of error covariance matrix
        P_k = Phi_k @ P_k @ Phi_k.T + Q_k

    # Make the first position the reference position
    qOut = quat.q_mult(qOut, quat.q_inv(qOut[0]))

    return qOut

class Madgwick:
    '''Madgwick's gradient descent filter.

        Parameters
        ----------
        rate : double
            sample rate [Hz]
        Beta : double
            algorithm gain
        Quaternion : array, shape (N,4)
            output quaternion describing the Earth relative to the sensor
        '''

    def __init__(self, rate=256.0, Beta=1.0, Quaternion=[1,0,0,0]):
        '''Initialization '''
        
        self.rate = rate
        self.SamplePeriod = 1/self.rate
        self.Beta = Beta
        self.Quaternion = Quaternion

    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        '''Calculate the best quaternion to the given measurement values.
        
        Parameters
        ----------
        Gyroscope : array, shape (,3)
            Angular velocity [rad/s]
        Accelerometer : array, shape (,3)
            Linear acceleration (Only the direction is used, so units don't matter.)
        Magnetometer : array, shape (,3)
            Orientation of local magenetic field.
            (Again, only the direction is used, so units don't matter.)
            
        '''

        q = self.Quaternion; # short name local variable for readability

        # Reference direction of Earth's magnetic field
        h = vector.rotate_vector(Magnetometer, q)
        b = np.hstack((0, np.sqrt(h[0]**2+h[1]**2), 0, h[2]))

        # Gradient decent algorithm corrective step
        F = [2*(q[1]*q[3] - q[0]*q[2])   - Accelerometer[0],
             2*(q[0]*q[1] + q[2]*q[3])   - Accelerometer[1],
             2*(0.5 - q[1]**2 - q[2]**2) - Accelerometer[2],
             2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2])   - Magnetometer[0],
             2*b[1]*(q[1]*q[2] - q[0]*q[3])   + 2*b[3]*(q[0]*q[1] + q[2]*q[3])   - Magnetometer[1],
             2*b[1]*(q[0]*q[2] + q[1]*q[3])   + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - Magnetometer[2]]

        J = np.array([
            [-2*q[2],                 	2*q[3],                    -2*q[0],                         2*q[1]],
            [ 2*q[1],                 	2*q[0],                	    2*q[3],                         2*q[2]],
            [0,                        -4*q[1],                    -4*q[2],                         0],
            [-2*b[3]*q[2],              2*b[3]*q[3],               -4*b[1]*q[2]-2*b[3]*q[0],       -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1],	2*b[1]*q[2]+2*b[3]*q[0],    2*b[1]*q[1]+2*b[3]*q[3],       -2*b[1]*q[0]+2*b[3]*q[2]],
            [ 2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1],    2*b[1]*q[0]-4*b[3]*q[2],        2*b[1]*q[1]]])

        step = J.T.dot(F)
        step = vector.normalize(step)	# normalise step magnitude

        # Compute rate of change of quaternion
        qDot = 0.5 * quat.q_mult(q, np.hstack([0, Gyroscope])) - self.Beta * step

        # Integrate to yield quaternion
        q = q + qDot * self.SamplePeriod
        self.Quaternion = vector.normalize(q).flatten()

class Mahony:
    '''Madgwick's implementation of Mayhony's AHRS algorithm

        Parameters
        ----------
        rate : double
            sample rate [Hz]
        Kp : algorithm proportional gain
        Ki : algorithm integral gain
        Quaternion : output quaternion describing the Earth relative to the sensor
    '''
    def __init__(self, rate=256.0, Kp=1.0, Ki=0, Quaternion=[1,0,0,0]):
        '''Initialization '''
        
        self.rate = rate
        self.SamplePeriod = 1/self.rate
        self.Kp = Kp
        self.Ki = Ki
        self.Quaternion = Quaternion
        self._eInt = [0, 0, 0]  # integral error

    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        '''Calculate the best quaternion to the given measurement values.
        
        Parameters
        ----------
        Gyroscope : array, shape (,3)
            Angular velocity [rad/s]
        Accelerometer : array, shape (,3)
            Linear acceleration (Only the direction is used, so units don't
            matter.)
        Magnetometer : array, shape (,3)
            Orientation of local magenetic field.
            (Again, only the direction is used, so units don't matter.)
            
        '''

        q = self.Quaternion; # short name local variable for readability

        # Reference direction of Earth's magnetic field
        h = vector.rotate_vector(Magnetometer, q)
        b = np.hstack((0, np.sqrt(h[0]**2+h[1]**2), 0, h[2]))

        # Estimated direction of gravity and magnetic field
        v = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2])

        w = np.array([
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]),
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]),
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2)]) 

        # Error is sum of cross product between estimated direction and measured
        # direction of fields
        e = np.cross(Accelerometer, v) + np.cross(Magnetometer, w) 

        if self.Ki > 0:
            self._eInt += e * self.SamplePeriod  
        else:
            self._eInt = np.array([0, 0, 0], dtype=float)

        # Apply feedback terms
        Gyroscope += self.Kp * e + self.Ki * self._eInt;            

        # Compute rate of change of quaternion
        qDot = 0.5 * quat.q_mult(q, np.hstack([0, Gyroscope])).flatten()

        # Integrate to yield quaternion
        q += qDot * self.SamplePeriod

        self.Quaternion = vector.normalize(q)



if __name__ == '__main__':
    '''
    in_file = r'../others/skinematics-invalid-sqrt.txt'
    import pandas as pd
    
    data = pd.read_csv(in_file, delimiter='\t')
    omega = data.filter(regex='Gyr*').values
    acc = data.filter(regex='Acc*').values    
    analytical(omega = omega, accMeasured=acc)
    
    '''
    from sensors.xsens import XSens

    in_file = r'tests/data/data_xsens.txt'
    initial_orientation = np.array([ [1,0,0],
                                     [0,0,-1],
                                     [0,1,0]])

    #in_file = r'tests/data/data_xsens2.txt'
    #initial_orientation = np.array([[0,0,-1],
                                    #[1, 0, 0],
                                    #[0,-1,0]])
    initial_position = np.r_[0,0,0]

    sensor = XSens(in_file=in_file, R_init=initial_orientation,
                   pos_init=initial_position)

    # By default, the orientation quaternion gets automatically calculated,
    # using "analytical"
    q_analytical = sensor.quat


    def show_result(imu_data):
        "Dummy function, to simplify the visualization"
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(imu_data.omega)
        axs[0].set_ylabel('Omega')
        axs[0].set_title(imu_data.q_type)
        axs[1].plot(imu_data.acc)
        axs[1].set_ylabel('Acc')
        axs[2].plot(imu_data.quat[:,1:])
        axs[2].set_ylabel('Quat')
        plt.show()    
        
    show_result(sensor)
    
    # Automatic re-calculation of orientation if "q_type" is changed
    sensor.set_qtype('madgwick')
    q_Madgwick = sensor.quat

    show_result(sensor)

    sensor.set_qtype('kalman')
    q_Kalman = sensor.quat

    show_result(sensor)

    print('Done')
