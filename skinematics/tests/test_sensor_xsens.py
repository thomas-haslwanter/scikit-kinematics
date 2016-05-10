import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))
sys.path.append('.')

import unittest
import imus
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    
    def test_import_xsens(self):
        # Get data, with a specified input from an XSens system
        inFile = r'data\data_xsens.txt'
        data = imus.import_data(inFile, type='XSens', paramList=['rate', 'acc', 'gyr'])
        rate = data[0]
        acc = data[1]
        omega = data[2]
        
        self.assertEqual(rate, 50.)
        self.assertAlmostEqual( (omega[0,2] - 0.050860000000000002), 0)
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
