import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import unittest

class TestSequenceFunctions(unittest.TestCase):
    
    def test_skinematics(self):
        for module in ['imus', 'markers', 'quat', 'rotmat', 'vector', 'viewer']:
            print(dir(module))
