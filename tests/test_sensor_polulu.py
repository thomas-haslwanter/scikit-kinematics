''' Test import data saved with Polulu-sensors, through subclassing "IMU_Base" '''

# Author: Thomas Haslwanter

import os
import unittest
from skinematics import imus
from time import sleep
from skinematics.sensors.polulu import Polulu

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_polulu(self):
        # Get data, with a specified input from a Polulu system
        in_data = {'rate':125}
        in_file = os.path.join('.', 'data', 'data_polulu.txt')
        sensor = Polulu(in_file=in_file, in_data = in_data, q_type=None)
        
        rate = sensor.rate
        acc = sensor.acc
        omega = sensor.omega
        
        self.assertAlmostEqual((rate - 125), 0)
        self.assertAlmostEqual( (omega[0,2] + 0.02596177), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Done')
    sleep(2)
