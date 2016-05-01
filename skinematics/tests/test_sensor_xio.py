
import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))
sys.path.append('.')

import unittest
import imus
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_xio(self):
        # Get data, with a specified input from an XIO system
        inFile = r'data\data_xio\00033_CalInertialAndMag.csv'
        data = imus.import_data(inFile, type='xio', paramList=['rate', 'acc', 'gyr', 'mag'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertAlmostEqual((rate - 256), 0)
        self.assertAlmostEqual( (omega[0,2] -10.125), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
