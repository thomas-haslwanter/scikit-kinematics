
import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))
sys.path.append('.')

import unittest
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, array, r_, vstack, abs, tile, pi
from numpy.linalg import norm
import imus, quat, vector, rotmat
from time import sleep

import import_data

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
        inFile = 'data_xsens.txt'
        data = import_data.XSens(inFile, ['Counter', 'Acc', 'Gyr'])
        rate = data[0]
        acc = data[2]
        omega = data[3]
        
        initialPosition = array([0,0,0])
        R_initialOrientation = rotmat.R1(90)
        
        q1, pos1 = imus.calc_QPos(R_initialOrientation, omega, initialPosition, acc, rate)
        plt.plot(q1)
        plt.show()
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
