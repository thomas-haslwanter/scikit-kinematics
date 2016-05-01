import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))

import rotmat
import unittest
import numpy as np

class TestSequenceFunctions(unittest.TestCase):
    def test_R1(self):
        R1 = np.array([[1,0,0],
                    [0, np.sqrt(2)/2, -np.sqrt(2)/2],
                    [0, np.sqrt(2)/2,  np.sqrt(2)/2]])
        
        self.assertTrue(np.all(np.abs(R1 - rotmat.R1(45))<1e-5))
        
    def test_R2(self):
        R2 = np.array([[ np.sqrt(2)/2, 0, np.sqrt(2)/2],
                       [0, 1, 0],
                       [-np.sqrt(2)/2, 0, np.sqrt(2)/2]])
        
        self.assertTrue(np.all(np.abs(R2 - rotmat.R2(45))<1e-5))
        
    def test_R3(self):
        R3 = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        
        self.assertTrue(np.all(np.abs(R3 - rotmat.R3(45))<1e-5))

    def test_symbolic(self):
        R_Fick = rotmat.R3_s()*rotmat.R2_s()*rotmat.R1_s()
        
    def test_Fick(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Fick = rotmat.rotmat2Fick(testmat)
        correct = np.r_[[0,0,np.pi/4]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Fick)), 0)
    
    def test_Helmholtz(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Helm = rotmat.rotmat2Helmholtz(testmat)
        correct = np.r_[[0,0,np.pi/4]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Helm)), 0)
        
    def test_Euler(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Euler = rotmat.rotmat2Euler(testmat)
        correct = np.r_[[np.pi/4,0,0]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Euler)), 0)
        
if __name__ == '__main__':
    unittest.main()
