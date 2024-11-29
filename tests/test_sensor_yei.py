''' Test import data saved with YEI-sensors, through subclassing "IMU_Base" '''

# Author: Thomas Haslwanter

import os
import unittest
from skinematics import imus
from time import sleep
from skinematics.sensors.yei import YEI

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_yei(self):
        # Get data, with a specified input from a YEI system
        in_file = os.path.join('.', 'data', 'data_yei.txt')
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
