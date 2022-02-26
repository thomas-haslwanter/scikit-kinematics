"""
Test import from import data saved with Vicon-sensors, through subclassing 'IMU_Base'
"""

# Author: Thomas Haslwanter

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', 'src','skinematics'))

import unittest
import imus
from time import sleep
from sensors.vicon import Vicon

class TestSequenceFunctions(unittest.TestCase):

    def test_import_vicon(self):
        # Get data, with a specified input from a Vicon system
        in_file = os.path.join(myPath, 'data', 'LeftFoot-marche01.c3d')

        sensor = Vicon(in_file=in_file, q_type=None)
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega

        self.assertEqual(rate, 50.)
        self.assertAlmostEqual( (omega[0,2] - 0.0447508), 0)

    def test_IMU_vicon(self):
        # Get data, with a specified input from a Vicon system
        in_file = os.path.join(myPath, 'data', 'LeftFoot-marche01.c3d')
        my_IMU = Vicon(in_file=in_file)

        self.assertEqual(my_IMU.omega.size, 48744)

if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
