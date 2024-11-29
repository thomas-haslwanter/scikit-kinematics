""" Test import from import data saved with XIO3 sensors from x-io,
through subclassing 'IMU_Base' """

# Author: Thomas Haslwanter

import numpy as np
import os

import unittest
from time import sleep
from skinematics.sensors.xio3 import XIO3

myPath = os.path.dirname(os.path.abspath(__file__))

class TestSequenceFunctions(unittest.TestCase):

    def test_import_xio(self):
        # Get data, with a specified input from an XIO system
        in_file = os.path.join(myPath, 'data', 'data_xio3')
        sensor = XIO3(in_file=in_file, q_type=None, in_data=50.)

        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega

        self.assertAlmostEqual((rate - 50), 0)
        self.assertAlmostEqual(
            (np.rad2deg(omega[0, 2]) + 0.014149103923330339), 0)

if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(0.2)
