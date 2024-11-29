import unittest
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, array, r_, vstack, abs, tile, pi
from numpy.linalg import norm
from skinematics import markers, quat, vector, rotmat
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.qz  = r_[cos(0.1), 0,0,sin(0.1)]
        self.qy  = r_[cos(0.1),0,sin(0.1), 0]

        self.quatMat = vstack((self.qz,self.qy))

        self.q3x = r_[sin(0.1), 0, 0]
        self.q3y = r_[2, 0, sin(0.1), 0]

        self.delta = 1e-4
        
    def test_analyze3dmarkers(self):
        # Test-data Generation:
        
        # 1) Three marker locations
        M = np.empty((3,3))
        M[0] = np.r_[0,0,0]
        M[1]= np.r_[1,0,0]
        M[2] = np.r_[1,1,0]
        M -= np.mean(M, axis=0) 
    
        # 2) a gradual translation from 0 to [1,1,0], over 10 sec
        t = np.arange(0,10,0.1)
        translation = (np.c_[[1,1,0]]*t).T
    
        # 3) A rotation with 100 deg/s about the z-axis
        q = np.vstack((np.zeros_like(t), np.zeros_like(t),quat.deg2quat(100*t))).T
    
        # Calculate the location of the three test-markers
        M0 = vector.rotate_vector(M[0], q) + translation
        M1 = vector.rotate_vector(M[1], q) + translation
        M2 = vector.rotate_vector(M[2], q) + translation
    
        data = np.hstack((M0,M1,M2))
    
        (pos, ori) = markers.analyze_3Dmarkers(data, data[0])
        
        self.assertAlmostEqual(np.max(np.abs(pos-translation)), 0)
        self.assertAlmostEqual(np.max(np.abs(ori-q)), 0)
    
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
