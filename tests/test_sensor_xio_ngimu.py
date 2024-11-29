""" Test import from import data saved with NGIMU sensors from x-io,
through subclassing 'IMU_Base' """

# Author: Thomas Haslwanter

import numpy as np
import os

import unittest
from time import sleep
from skinematics import imus
from skinematics.sensors.xio_ngimu import NGIMU

class TestSequenceFunctions(unittest.TestCase):

    def test_import_xio(self):
        # Get data, with a specified input from an XIO system
        in_file = os.path.join('.', 'data', 'data_ngimu')
        sensor = NGIMU(in_file=in_file, q_type=None)

        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega

        self.assertAlmostEqual((rate - 50), 0)
        self.assertAlmostEqual( (np.rad2deg(omega[0,2]) + 0.0020045), 0)

if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(0.2)
