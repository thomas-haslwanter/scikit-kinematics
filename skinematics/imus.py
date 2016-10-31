'''
Utilities for movements recordings with inertial measurement units (IMUs)
Currently data from the following systems are supported

    - XIO
    - XSens
    - YEI
    - Polulu
'''

'''
Author: Thomas Haslwanter
Version: 1.7
Date: Oct-2016
'''

import numpy as np
import scipy as sp
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import pandas as pd 
from numpy import r_, sum
import re

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from skinematics import quat, vector, misc
import easygui

class IMU:
    '''
    Class for working with working with inertial measurement units (IMUs)
    
    An IMU object can be initialized
        - by giving a filename
        - by providing measurement data and a samplerate
        - without giving any parameter; in that case the user is prompted
          to select an infile

    Parameters
    ----------
    inFile : string
        path- and file-name of data file
    inType : string
        Description of the type of the in-file. Has to be one of the following:
            - xio
            - XSens
            - yei
            - polulu
    inData : dictionary
        The following fields are required:

        acc : (N x 3) array
             Linear acceleration [m/s^2], in the x-, y-, and z-direction
        omega : (N x 3) array
             Angular velocity [deg/s], about the x-, y-, and z-axis
        [mag] : (N x 3) array (optional)
             Local magnetic field, in the x-, y-, and z-direction
        rate: integer
            sample rate [Hz]

    Notes
    -----
    
    IMU-Properties:
        - source
        - acc
        - omega
        - mag
        - rate
        - totalSamples
        - duration
        - dataType

    Examples
    --------
	>>> myimu = IMU(r'tests/data/data_xsens.txt', inType='XSens')
	>>> 
	>>> initialOrientation = np.array([[1,0,0],
	>>>                                [0,0,-1],
	>>>                                [0,1,0]])
	>>> initialPosition = np.r_[0,0,0]
	>>> 
	>>> myimu.calc_orientation(initialOrientation)
	>>> q_simple = myimu.quat[:,1:]
	>>> 
	>>> calcType = 'Madgwick'
	>>> myimu.calc_orientation(initialOrientation, type=calcType)
	>>> q_Kalman = myimu.quat[:,1:]
 
    '''


    def __init__(self, inFile = None, inType='XSens', inData = None):
        '''Initialize an IMU-object'''

        if inData is not None:
            self.setData(inData)
        else: 
            if inFile is None:
                inFile = self._selectInput()
            if os.path.exists(inFile):    
                self.source = inFile

                try:
                    data = import_data(inFile=self.source, type=inType, paramList=['rate', 'acc', 'omega', 'mag'])
                    self.rate = data[0]
                    self.acc= data[1]
                    self.omega = data[2]
                    self.mag = data[3]
                    self._setInfo()

                    print('data read in!')
                except IOError:
                    print('Could not read ' + inFile)
            else:
                print(inFile + ' does NOT exist!')

    def setData(self, data):
        ''' Set the properties of an IMU-object. '''

        if 'rate' not in data.keys():
            print('Set the "rate" to the default value (100 Hz).')
            data['rate'] = 100.0

        self.rate = data['rate']
        self.acc= data['acc']
        self.omega = data['omega']
        if 'mag' in data.keys():
            self.mag = data['mag']
        self.source = None
        self._setInfo()

    def calc_orientation(self, R_initialOrientation, type='quatInt'):
        '''
        Calculate the current orientation

        Parameters
        ----------
        R_initialOrientation : 3x3 array
                approximate alignment of sensor-CS with space-fixed CS
        type : string
                - 'quatInt' .... quaternion integration of angular velocity
                - 'Kalman' ..... quaternion Kalman filter
                - 'Madgwick' ... gradient descent method, efficient
                - 'Mahony' ....  formula from Mahony, as implemented by Madgwick

        '''

        initialPosition = np.r_[0,0,0]

        if type == 'quatInt':
            (quaternion, position) = calc_QPos(R_initialOrientation, self.omega, initialPosition, self.acc, self.rate)

        elif type == 'Kalman':
            self._checkRequirements()
            quaternion =  kalman_quat(self.rate, self.acc, self.omega, self.mag)

        elif type == 'Madgwick':
            self._checkRequirements()
                    
            # Initialize object
            AHRS = MadgwickAHRS(SamplePeriod=1./self.rate, Beta=0.1)
            quaternion = np.zeros((self.totalSamples, 4))
            
            # The "Update"-function uses angular velocity in radian/s, and only the directions of acceleration and magnetic field
            Gyr = self.omega
            Acc = vector.normalize(self.acc)
            Mag = vector.normalize(self.mag)
            
            #for t in range(len(time)):
            for t in misc.progressbar(range(self.totalSamples), 'Calculating the Quaternions ', 25) :
                AHRS.Update(Gyr[t], Acc[t], Mag[t])
                quaternion[t] = AHRS.Quaternion
                
        elif type == 'Mahony':
            self._checkRequirements()
                    
            # Initialize object
            AHRS = MahonyAHRS(SamplePeriod=1./self.rate, Kp=0.5)
            quaternion = np.zeros((self.totalSamples, 4))
            
            # The "Update"-function uses angular velocity in radian/s, and only the directions of acceleration and magnetic field
            Gyr = self.omega
            Acc = vector.normalize(self.acc)
            Mag = vector.normalize(self.mag)
            
            #for t in range(len(time)):
            for t in misc.progressbar(range(self.totalSamples), 'Calculating the Quaternions ', 25) :
                AHRS.Update(Gyr[t], Acc[t], Mag[t])
                quaternion[t] = AHRS.Quaternion
            
        else:
            print('Unknown orientation type: {0}'.format(type))
            return

        self.quat = quaternion

    def calc_position(self, initialPosition):
        '''Calculate the position, assuming that the orientation is already known.'''

        # Acceleration, velocity, and position ----------------------------
        # From q and the measured acceleration, get the \frac{d^2x}{dt^2}
        g = sp.constants.g
        g_v = np.r_[0, 0, g] 
        accReSensor = self.acc - vector.rotate_vector(g_v, quat.quatinv(self.quat))
        accReSpace = vector.rotate_vector(accReSensor, self.quat)

        # Position and Velocity through integration, assuming 0-velocity at t=0
        vel = np.nan*np.ones_like(accReSpace)
        pos = np.nan*np.ones_like(accReSpace)

        for ii in range(accReSpace.shape[1]):
            vel[:,ii] = cumtrapz(accReSpace[:,ii], dx=1./self.rate, initial=0)
            pos[:,ii] = cumtrapz(vel[:,ii],        dx=1./self.rate, initial=initialPosition[ii])

        self.pos = pos

    def _checkRequirements(self):
        '''Check if all the necessary variables are available.'''
        required = [ 'rate', 'acc', 'omega', 'mag' ]
    
        for field in required:
            if field not in vars(self):
                print('Cannot find {0} in calc_orientation!'.format(field))
            
    def _selectInput(self):
        '''GUI for the selection of an in-file. '''

        fullInFile = easygui.fileopenbox(msg='Input data: ', title='Selection', default='*.txt')
        print('Selection: ' + fullInFile)
        return fullInFile

    def _setInfo(self):
        '''Set the information properties of that IMU'''

        self.totalSamples = len(self.omega)
        self.duration = np.float(self.totalSamples)/self.rate # [sec]
        self.dataType = str(self.omega.dtype)

def import_data(inFile=None, type='XSens', paramList=['rate', 'acc', 'omega', 'mag']):
    '''
    Read in Rate and stored 3D parameters from IMUs

    Parameters
    ----------
    inFile : string
             Path and name of input file
    type : sensor-type. Has to be either ['XSens', 'xio', 'yei', 'polulu']
    paramList: list of strings
               You can select between ['rate', 'acc', 'omega', 'mag', 'others']

    Returns
    -------
    List, containing
    rate : float
        Sampling rate
    [List of x/y/z Parameter Values]

    Examples
    --------
    >>> data = import_data(fullInFile, type='XSens', paramList=['rate', 'acc', 'omega'])
    >>> rate = data[0]
    >>> accValues = data[1]
    >>> Omega = data[2]
    
    '''

    if inFile is None:
        inFile = easygui.fileopenbox(msg='Please select an in-file containing XSens-IMU data: ', title='Data-selection', default='*.txt')

    varList = ['acc', 'omega', 'mag', 'rate', 'others']
    
    dataDict = {}
    for var in varList:
        dataDict[var]=None
    
    if type == 'XSens':
        from skinematics.sensors import xsens
        data = xsens.get_data(inFile)
    elif type == 'xio':
        from skinematics.sensors import xio
        data = xio.get_data(inFile)
    elif type == 'yei':
        from skinematics.sensors import yei
        data = yei.get_data(inFile)
    elif type == 'polulu':
        from skinematics.sensors import polulu
        data = polulu.get_data(inFile)
    else:
        raise ValueError
        
    dataDict['rate'] = data[0]
    dataDict['acc']  = data[1]
    dataDict['omega']  = data[2]
    dataDict['mag']  = data[3]
    
    returnValues = []
    
    # By default, return all values, in alphabetical order
    if paramList == []:
        paramList = list(dataDict.keys())
        paramList.sort()
        print('Return-values: {0}'.format(paramList))
        
    for param in paramList:
        try:
            returnValues.append(dataDict[param])
        except KeyError as err:
            print('Please check the parameter {0}'.format(param))
            print(err)
            
    return returnValues
        
        
def _read_xsens(inFile):
    '''Read data from an XSens sensor.
    The data returned are (in that order): [rate, acceleration, angular_velocity, mag_field_direction]'''
    
    # Get the sampling rate from the second line in the file
    try:
        fh = open(inFile)
        fh.readline()
        line = fh.readline()
        rate = np.float(line.split(':')[1].split('H')[0])
        fh.close()
        returnValues = [rate]

    except FileNotFoundError:
        print('{0} does not exist!'.format(inFile))
        return -1

    # Read the data
    data = pd.read_csv(inFile,
                       sep='\t',
                       skiprows=4, 
                       index_col=False)

    # Extract the columns that you want, by name
    paramList=['Acc', 'Gyr', 'Mag']
    for param in paramList:
        Expression = param + '*'
        returnValues.append(data.filter(regex=Expression).values)

    return returnValues

def _read_xio(inFile):
    '''Read data from an XIO sensor.
    The data returned are (in that order): [rate, acceleration, angular_velocity, mag_field_direction, packet_nr]'''
    
    # XIO-definition of the sampling rate 
    # Get the "Register"
    inFile = '_'.join(inFile.split('_')[:-1]) + '_Registers.csv'
    
    # Make sure that we read in the correct one
    inFile = '_'.join(inFile.split('_')[:-1]) + '_CalInertialAndMag.csv'
    data = pd.read_csv(inFile)
    
    # Generate a simple list of column names
    rate = 256  # currently hardcoded
    
    returnValues = [rate]
    
    # Extract the columns that you want, by name
    paramList=['Acc', 'Gyr', 'Mag', 'Packet']
    for Expression in paramList:
        returnValues.append(data.filter(regex=Expression).values)
        
    return returnValues    

def _read_yei(inFile):
    '''Read data from an YEI sensor.
    The data returned are (in that order): [rate, acceleration, angular_velocity, mag_field_direction]'''
    
    data = pd.read_csv(inFile)
    
    # Generate a simple list of column names
    newColumns = []
    pattern = re.compile('.*%(\w+)\((\w+)\)')
    for name in data.columns:
        newColumns.append(pattern.match(name).groups()[1])
    data.columns = newColumns
    
    
    # Calculate rate (ChipTime is in microsec)
    start = data.ChipTimeUS[0] * 1e-6    # microseconds to seconds
    stop = data.ChipTimeUS.values[-1] * 1e-6    # pandas can't count backwards
    rate = len(data) / (stop-start)
    
    returnValues = [rate]
    
    # Extract the columns that you want, by name
    paramList=['Accel', 'Gyro', 'Compass']
    for Expression in paramList:
        returnValues.append(data.filter(regex=Expression).values)
        
    return returnValues

    
def calc_QPos(R_initialOrientation, omega, initialPosition, accMeasured, rate):
    ''' Reconstruct position and orientation, from angular velocity and linear acceleration.
    Assumes a start in a stationary position. No compensation for drift.

    Parameters
    ----------
    omega : ndarray(N,3)
        Angular velocity, in [rad/s]
    accMeasured : ndarray(N,3)
        Linear acceleration, in [m/s^2]
    initialPosition : ndarray(3,)
        initial Position, in [m]
    R_initialOrientation: ndarray(3,3)
        Rotation matrix describing the initial orientation of the sensor,
        except a mis-orienation with respect to gravity
    rate : float
        sampling rate, in [Hz]

    Returns
    -------
    q : ndarray(N,3)
        Orientation, expressed as a quaternion vector
    pos : ndarray(N,3)
        Position in space [m]

    Example
    -------
    >>> q1, pos1 = calc_QPos(R_initialOrientation, omega, initialPosition, acc, rate)

    '''

    # Transform recordings to angVel/acceleration in space --------------

    # Orientation of \vec{g} with the sensor in the "R_initialOrientation"
    g = 9.81
    g0 = np.linalg.inv(R_initialOrientation).dot(r_[0,0,g])

    # for the remaining deviation, assume the shortest rotation to there
    q0 = vector.qrotate(accMeasured[0], g0)    
    R0 = quat.quat2rotmat(q0)

    # combine the two, to form a reference orientation. Note that the sequence
    # is very important!
    R_ref = R_initialOrientation.dot(R0)
    q_ref = quat.rotmat2quat(R_ref)

    # Calculate orientation q by "integrating" omega -----------------
    q = quat.vel2quat(omega, q_ref, rate, 'bf')

    # Acceleration, velocity, and position ----------------------------
    # From q and the measured acceleration, get the \frac{d^2x}{dt^2}
    g_v = r_[0, 0, g] 
    accReSensor = accMeasured - vector.rotate_vector(g_v, quat.quatinv(q))
    accReSpace = vector.rotate_vector(accReSensor, q)

    # Make the first position the reference position
    q = quat.quatmult(q, quat.quatinv(q[0]))

    # compensate for drift
    #drift = np.mean(accReSpace, 0)
    #accReSpace -= drift*0.7

    # Position and Velocity through integration, assuming 0-velocity at t=0
    vel = np.nan*np.ones_like(accReSpace)
    pos = np.nan*np.ones_like(accReSpace)

    for ii in range(accReSpace.shape[1]):
        vel[:,ii] = cumtrapz(accReSpace[:,ii], dx=1./rate, initial=0)
        pos[:,ii] = cumtrapz(vel[:,ii],        dx=1./rate, initial=initialPosition[ii])

    return (q, pos)

def kalman_quat(rate, acc, omega, mag):
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

    Returns
    -------
    qOut : (N,4) ndarray
    	   unit quaternion, describing the orientation relativ to the coordinate system spanned by the local magnetic field, and gravity

    Notes
    -----
    Based on "Design, Implementation, and Experimental Results of a Quaternion-Based Kalman Filter for Human Body Motion Tracking" Yun, X. and Bachman, E.R., IEEE TRANSACTIONS ON ROBOTICS, VOL. 22, 1216-1227 (2006)

    '''

    numData = len(acc)

    # Set parameters for Kalman Filter
    tstep = 1/rate
    tau = [0.5, 0.5, 0.5]	# from Yun, 2006

    # Initializations 
    x_k = np.zeros(7)	# state vector
    z_k = np.zeros(7)   # measurement vector
    z_k_pre = np.zeros(7)
    P_k = np.matrix( np.eye(7) )		 # error covariance matrix P_k

    Phi_k = np.matrix( np.zeros((7,7)) ) # discrete state transition matrix Phi_k
    for ii in range(3):
        Phi_k[ii,ii] = np.exp(-tstep/tau[ii])

    H_k = np.matrix( np.eye(7) )		# Identity matrix

    Q_k = np.matrix( np.zeros((7,7)) )	# process noise matrix Q_k
    D = 0.0001*r_[0.4, 0.4, 0.4]		# [rad^2/sec^2]; from Yun, 2006
                                                                            # check 0.0001 in Yun
    for ii in range(3):
        Q_k[ii,ii] =  D[ii]/(2*tau[ii])  * ( 1-np.exp(-2*tstep/tau[ii]) )

    # Evaluate measurement noise covariance matrix R_k
    R_k = np.matrix( np.zeros((7,7)) )
    r_angvel = 0.01;      # [rad**2/sec**2]; from Yun, 2006
    r_quats = 0.0001;     # from Yun, 2006
    for ii in range(7):
        if ii<3:
            R_k[ii,ii] = r_angvel
        else:
            R_k[ii,ii] = r_quats

    # Calculation of orientation for every time step
    qOut = np.zeros( (numData,4) )

    for ii in range(numData):
        accelVec  = acc[ii,:]
        magVec    = mag[ii,:]
        angvelVec = omega[ii,:]
        z_k_pre = z_k.copy()	# watch out: by default, Python passes the reference!!

        # Evaluate quaternion based on acceleration and magnetic field data
        accelVec_n = vector.normalize(accelVec)
        magVec_hor = magVec - accelVec_n * accelVec_n.dot(magVec)
        magVec_n   = vector.normalize(magVec_hor)
        basisVectors = np.vstack( (magVec_n, np.cross(accelVec_n, magVec_n), accelVec_n) ).T
        quatRef = quat.quatinv(quat.rotmat2quat(basisVectors)).flatten()

        # Update measurement vector z_k
        z_k[:3] = angvelVec
        z_k[3:] = quatRef

        # Calculate Kalman Gain
        # K_k = P_k * H_k.T * inv(H_k*P_k*H_k.T + R_k)
        # Check: why is H_k used in the original formulas?
        K_k = P_k * np.linalg.inv(P_k + R_k)

        # Update state vector x_k
        x_k += np.array( K_k.dot(z_k-z_k_pre) ).ravel()

        # Evaluate discrete state transition matrix Phi_k
        Phi_k[3,:] = r_[-x_k[4]*tstep/2, -x_k[5]*tstep/2, -x_k[6]*tstep/2,              1, -x_k[0]*tstep/2, -x_k[1]*tstep/2, -x_k[2]*tstep/2]
        Phi_k[4,:] = r_[ x_k[3]*tstep/2, -x_k[6]*tstep/2,  x_k[5]*tstep/2, x_k[0]*tstep/2,               1,  x_k[2]*tstep/2, -x_k[1]*tstep/2]
        Phi_k[5,:] = r_[ x_k[6]*tstep/2,  x_k[3]*tstep/2, -x_k[4]*tstep/2, x_k[1]*tstep/2, -x_k[2]*tstep/2,               1,  x_k[0]*tstep/2]
        Phi_k[6,:] = r_[-x_k[5]*tstep/2,  x_k[4]*tstep/2,  x_k[3]*tstep/2, x_k[2]*tstep/2,  x_k[1]*tstep/2, -x_k[0]*tstep/2,               1]

        # Update error covariance matrix
        #P_k = (eye(7)-K_k*H_k)*P_k
        # Check: why is H_k used in the original formulas?
        P_k = (H_k - K_k) * P_k

        # Projection of state quaternions
        x_k[3:] += quat.quatmult(0.5*x_k[3:],r_[0, x_k[:3]]).flatten()
        x_k[3:] = vector.normalize( x_k[3:] )
        x_k[:3] = np.zeros(3)
        x_k[:3] = tstep * (-x_k[:3]+z_k[:3])

        qOut[ii,:] = x_k[3:]

        # Projection of error covariance matrix
        P_k = Phi_k * P_k * Phi_k.T + Q_k

    # Make the first position the reference position
    qOut = quat.quatmult(qOut, quat.quatinv(qOut[0]))
        
    return qOut

class MadgwickAHRS:
    '''Madgwick's gradient descent filter.
    '''
    
    def __init__(self, SamplePeriod=1./256, Beta=1.0, Quaternion=[1,0,0,0]):
        '''Initialization
        
        Parameters
        ----------
        SamplePeriod : sample period
        Beta : algorithm gain
        Quaternion : output quaternion describing the Earth relative to the sensor
        '''
        self.SamplePeriod = SamplePeriod
        self.Beta = Beta
        self.Quaternion = Quaternion
    
    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        '''Calculate the best quaternion to the given measurement values.'''
        
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
        qDot = 0.5 * quat.quatmult(q, np.hstack([0, Gyroscope])) - self.Beta * step

        # Integrate to yield quaternion
        q = q + qDot * self.SamplePeriod
        self.Quaternion = vector.normalize(q).flatten()
        
class MahonyAHRS:
    '''Madgwick's implementation of Mayhony's AHRS algorithm
    '''
    def __init__(self, SamplePeriod=1./256, Kp=1.0, Ki=0, Quaternion=[1,0,0,0]):
        '''Initialization
        
        Parameters
        ----------
        SamplePeriod : sample period
        Kp : algorithm proportional gain
        Ki : algorithm integral gain
        Quaternion : output quaternion describing the Earth relative to the sensor
        '''
        self.SamplePeriod = SamplePeriod
        self.Kp = Kp
        self.Ki = Ki
        self.Quaternion = Quaternion
        self._eInt = [0, 0, 0]  # integral error

    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        '''Calculate the best quaternion to the given measurement values.'''
        
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

        # Error is sum of cross product between estimated direction and measured direction of fields
        e = np.cross(Accelerometer, v) + np.cross(Magnetometer, w) 
        
        if self.Ki > 0:
            self._eInt += e * self.SamplePeriod  
        else:
            self._eInt = np.array([0, 0, 0], dtype=np.float)
        
        # Apply feedback terms
        Gyroscope += self.Kp * e + self.Ki * self._eInt;            
        
        # Compute rate of change of quaternion
        qDot = 0.5 * quat.quatmult(q, np.hstack([0, Gyroscope])).flatten()

        # Integrate to yield quaternion
        q += qDot * self.SamplePeriod

        self.Quaternion = vector.normalize(q)

if __name__ == '__main__':
    myIMU = IMU(inFile = r'tests/data/data_polulu.txt', inType='polulu')
    myIMU.calc_orientation(np.eye(3), type='Mahony')
    quat = myIMU.quat[:,1:]
    fig, axs = plt.subplots(3,1)
    axs[0].plot(myIMU.omega)
    axs[0].set_ylabel('Omega')
    axs[1].plot(myIMU.acc)
    axs[1].set_ylabel('Acc')
    axs[2].plot(quat)
    axs[2].set_ylabel('Quat')
    plt.show()
    
    
    '''
    myimu = IMU(r'tests/data/data_xsens.txt')

    initialOrientation = np.array([[1,0,0],
                                   [0,0,-1],
                                   [0,1,0]])

    myimu.calc_orientation(initialOrientation, type='quatInt')
    q_simple = myimu.quat[:,1:]
    
    #calcType = 'Mahony'
    calcType = 'Madgwick'
    calcType = 'Kalman'
    myimu.calc_orientation(initialOrientation, type=calcType)
    q_Kalman = myimu.quat[:,1:]
    
    #myimu.calc_position(initialPosition)

    t = np.arange(myimu.totalSamples)/myimu.rate
    plt.plot(t, q_simple, '-', label='simple')
    plt.hold(True)
    plt.plot(t, q_Kalman, '--', label=calcType)
    plt.legend()
    #plt.plot(t, myimu.pos)
    plt.show()
    '''

    
    print('Done')
