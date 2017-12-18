'''
Test import from import data saved with XIO-sensors, through subclassing "IMU_Base"
'''

'''
Author: Thomas Haslwanter
Version: 0.1
Date: Sept-2017
'''

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

import unittest
import imus
from time import sleep
from sensors.xio import XIO

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_xio(self):
        # Get data, with a specified input from an XIO system
        in_file = os.path.join(myPath, 'data', 'data_xio')
        sensor = XIO(in_file=in_file, q_type=None)
        
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega
        
        self.assertAlmostEqual((rate - 256), 0)
        self.assertAlmostEqual( (omega[0,2] -10.125), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(0.2)
