'''
Test import data saved with YEI-sensors, through subclassing "IMU_Base"
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
from skinematics import imus
from time import sleep
from sensors.yei import YEI

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_yei(self):
        # Get data, with a specified input from a YEI system
        in_file = os.path.join(myPath, 'data', 'data_yei.txt')
        sensor = YEI(in_file=in_file, q_type=None)
        
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega
        
        self.assertAlmostEqual((rate - 109.99508526563774), 0)
        self.assertAlmostEqual( (omega[0,2] - 0.0081446301192045212), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Done')
    sleep(2)
