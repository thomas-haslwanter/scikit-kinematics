import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', '..'))

import unittest

class TestSequenceFunctions(unittest.TestCase):
    
    def test_skinematics(self):
        for module in ['imus', 'markers', 'quat', 'rotmat', 'vector', 'viewer']:
            print(dir(module))
