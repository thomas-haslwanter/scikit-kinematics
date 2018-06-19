#from .context import skinematics
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', '..'))

import unittest
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, array, r_, vstack, abs, tile, pi
from numpy.linalg import norm
import os
from skinematics import imus, quat, vector, rotmat
from time import sleep
from skinematics.simulations.simulate_movements import simulate_imu

class TestSequenceFunctions(unittest.TestCase):
    
    def setUp(self):
        self.qz  = r_[cos(0.1), 0,0,sin(0.1)]
        self.qy  = r_[cos(0.1),0,sin(0.1), 0]

        self.quatMat = vstack((self.qz,self.qy))

        self.q3x = r_[sin(0.1), 0, 0]
        self.q3y = r_[2, 0, sin(0.1), 0]

        self.delta = 1e-4
        
       # Simulate IMU-data
        duration_movement = 1    # [sec]
        duration_total = 1      # [sec]
        rate = 100             # [Hz]
        
        B0 = vector.normalize([0, -1, -1])
        
        rotation_axis = [0, 1, 0]
        angle = 90
        
        translation = [1,0,0]
        distance = 0
        
        q_init = [0,0,0]
        pos_init = [0,0,0]
        
        self.imu_signals, self.body_pos_orient =  simulate_imu(rate, duration_movement, duration_total,
                    q_init = q_init, rotation_axis=rotation_axis, deg=angle,
                    pos_init = pos_init, direction=translation, distance=distance,
                    B0=B0) 

        
    def test_analytical(self):
        
        # Analyze the simulated data with "analytical"
        q, pos, vel = imus.analytical(R_initialOrientation=np.eye(3),
                         omega = self.imu_signals['omega'],
                         initialPosition=np.zeros(3),
                         accMeasured = self.imu_signals['gia'],
                         rate = rate)                         
        
        # and then check, if the position is [0,0,0], and the orientation-quat = [0, sin(45), 0]
        self.assertTrue(np.max(np.abs(pos[-1]))<0.001)      # less than 1mm
        
        result = quat.q_vector(q[-1])
        correct = array([ 0.,  np.sin(np.deg2rad(45)),  0.])
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        
    def test_kalman(self):
        
        # Analyze the simulated data with "kalman"
        imu = self.imu_signals
        q_kalman = kalman(imu.rate, imu.gia, imu.omega, imu.magnetic)
        
        # and then check, if the quat_vector = [0, sin(45), 0]
        result = quat.q_vector(q_kalman[-1])
        correct = array([ 0.,  np.sin(np.deg2rad(45)),  0.])
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        ## Get data
        #inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        #from skinematics.sensors.xsens import XSens
        
        #initialPosition = array([0,0,0])
        #R_initialOrientation = rotmat.R(0,90)
        
        #sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='kalman')
        #print(sensor.source)
        #q = sensor.quat
        
    def test_madgwick(self):
        
        ## Get data
        #inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        #from skinematics.sensors.xsens import XSens
        
        #initialPosition = array([0,0,0])
        #R_initialOrientation = rotmat.R(0,90)
        
        #sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='madgwick')
        #q = sensor.quat
        
    def test_mahony(self):
        
        ## Get data
        #inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        #from skinematics.sensors.xsens import XSens
        
        #initialPosition = array([0,0,0])
        #R_initialOrientation = rotmat.R(0,90)
        
        #sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='mahony')
        #q = sensor.quat
        
    def test_IMU_calc_orientation_position(self):
        """Currently, this only tests if the two functions are running through"""
        
        # Get data, with a specified input from an XSens system
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        initial_orientation = np.array([[1,0,0],
                                       [0,0,-1],
                                       [0,1,0]])
        initial_position = np.r_[0,0,0]
        
        from skinematics.sensors.xsens import XSens
        sensor = XSens(in_file=inFile, R_init=initial_orientation, pos_init=initial_position)
        sensor.calc_position()
        print('done')
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSequenceFunctions(methodName='test_IMU_calc_orientation_position'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    #unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(0.2)
