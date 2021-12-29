"""
Test import from import data saved with XSens-sensors, through subclassing 'IMU_Base'
"""

# Author: Thomas Haslwanter

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', 'src','skinematics'))

import unittest
import imus
from time import sleep
from sensors.xsens import XSens

class TestSequenceFunctions(unittest.TestCase):

    def test_import_xsens(self):
        # Get data, with a specified input from an XSens system
        in_file = os.path.join(myPath, 'data', 'data_xsens.txt')

        sensor = XSens(in_file=in_file, q_type=None)
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega

        self.assertEqual(rate, 50.)
        self.assertAlmostEqual( (omega[0,2] - 0.050860000000000002), 0)

    def test_IMU_xsens(self):
        # Get data, with a specified input from an XSens system
        in_file = os.path.join(myPath, 'data', 'data_xsens.txt')
        my_IMU = XSens(in_file=in_file)

        self.assertEqual(my_IMU.omega.size, 2859)

if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
