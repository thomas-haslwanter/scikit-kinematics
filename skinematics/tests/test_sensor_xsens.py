import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', '..'))

import unittest
from skinematics import imus
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_xsens(self):
        # Get data, with a specified input from an XSens system
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        data = imus.import_data(inFile, inType='XSens', paramList=['rate', 'acc', 'omega'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertEqual(rate, 50.)
        self.assertAlmostEqual( (omega[0,2] - 0.050860000000000002), 0)
        
    def test_IMU_xsens(self):
        # Get data, with a specified input from an XSens system
        inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        my_IMU = imus.IMU(inFile, inType='XSens')
        
        self.assertEqual(my_IMU.omega.size, 2859)
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
