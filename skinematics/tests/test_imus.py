
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, array, r_, vstack, abs, tile, pi
from numpy.linalg import norm
import imus, quat, vector, rotmat
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    
    def setUp(self):
        self.qz  = r_[cos(0.1), 0,0,sin(0.1)]
        self.qy  = r_[cos(0.1),0,sin(0.1), 0]

        self.quatMat = vstack((self.qz,self.qy))

        self.q3x = r_[sin(0.1), 0, 0]
        self.q3y = r_[2, 0, sin(0.1), 0]

        self.delta = 1e-4

    def test_calc_QPos(self):
        # Get data
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        data = imus.import_data(inFile, type='XSens', paramList=['rate', 'acc', 'omega', 'mag'])
        rate = data[0]
        acc = data[2]
        omega = data[3]
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R1(90)
        
        q1, pos1 = imus.calc_QPos(R_initialOrientation, omega, initialPosition, acc, rate)
        plt.plot(q1)
        plt.show()
        
    def test_import_empty(self):
        # Get data, with an empty input
        data = imus.import_data()
        
    def test_import_xsens(self):
        # Get data, with a specified input from an XSens system
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        data = imus.import_data(inFile, type='XSens', paramList=['rate', 'acc', 'omega'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertEqual(rate, 50.)
        self.assertAlmostEqual( (omega[0,2] - 0.050860000000000002), 0)

    def test_import_xio(self):
        # Get data, with a specified input from an XIO system
        inFile = os.path.join(myPath, 'data', 'data_xio', '00033_CalInertialAndMag.csv')
        data = imus.import_data(inFile, type='xio', paramList=['rate', 'acc', 'omega', 'mag'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertAlmostEqual((rate - 256), 0)
        self.assertAlmostEqual( (omega[0,2] -10.125), 0)
        
    def test_import_yei(self):
        # Get data, with a specified input from a YEI system
        inFile = os.path.join(myPath, 'data', 'data_yei.txt')
        data = imus.import_data(inFile, type='yei', paramList=['rate', 'acc', 'omega', 'mag'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertAlmostEqual((rate - 109.99508526563774), 0)
        self.assertAlmostEqual( (omega[0,2] - 0.0081446301192045212), 0)
        
    def test_import_polulu(self):
        # Get data, with a specified input from a POLULU system
        inFile = os.path.join(myPath, 'data', 'data_polulu.txt')
        data = imus.import_data(inFile, type='polulu', paramList=['rate', 'acc', 'omega', 'mag'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertAlmostEqual((rate - 125), 0)
        self.assertAlmostEqual( (acc[0,1] + 0.004575), 0)
        
    def test_IMU_calc_orientation_position(self):
        """Currently, this only tests if the two functions are running through"""
        
        # Get data, with a specified input from an XSens system
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        imu = imus.IMU(inFile)
        
        initial_orientation = np.array([[1,0,0],
                                       [0,0,-1],
                                       [0,1,0]])
        initial_position = np.r_[0,0,0]
        
        imu.calc_orientation(initial_orientation)
        imu.calc_position(initial_position)
        print('done')
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSequenceFunctions(methodName='test_IMU_calc_orientation_position'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    #unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
