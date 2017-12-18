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

class TestSequenceFunctions(unittest.TestCase):
    
    def setUp(self):
        self.qz  = r_[cos(0.1), 0,0,sin(0.1)]
        self.qy  = r_[cos(0.1),0,sin(0.1), 0]

        self.quatMat = vstack((self.qz,self.qy))

        self.q3x = r_[sin(0.1), 0, 0]
        self.q3y = r_[2, 0, sin(0.1), 0]

        self.delta = 1e-4

        '''
    def test_analytical(self):
        # Get data
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        from skinematics.sensors.xsens import XSens
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R(0,90)
        
        sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition)
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega
        
        plt.plot(sensor.quat)
        plt.show()
        '''
        
    def test_kalman(self):
        # Get data
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        from skinematics.sensors.xsens import XSens
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R(0,90)
        
        sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='kalman')
        print(sensor.source)
        q = sensor.quat
        
    def test_madgwick(self):
        # Get data
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        from skinematics.sensors.xsens import XSens
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R(0,90)
        
        sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='madgwick')
        q = sensor.quat
        
    def test_mahony(self):
        # Get data
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        from skinematics.sensors.xsens import XSens
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R(0,90)
        
        sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='mahony')
        q = sensor.quat
        
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
