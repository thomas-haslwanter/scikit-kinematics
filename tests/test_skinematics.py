import unittest
from importlib import import_module

class TestSequenceFunctions(unittest.TestCase):
    
    def test_skinematics(self):
        for module in ['imus', 'markers', 'quat', 'rotmat', 'vector', 'view']:
            print(dir(module))
            full_name = 'skinematics.' + module
            import_module(full_name)
