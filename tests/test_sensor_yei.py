
import sys
import os
sys.path.insert(0, os.path.abspath(r'..\skinematics'))
sys.path.append('.')

import unittest
import imus
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_yei(self):
        # Get data, with a specified input from a YEI system
        inFile = 'data\data_yei.txt'
        data = imus.import_data(inFile, type='yei', paramList=['rate', 'acc', 'gyr', 'mag'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertAlmostEqual((rate - 109.99508526563774), 0)
        self.assertAlmostEqual( (omega[0,2] - 0.0081446301192045212), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Done')
    sleep(2)
