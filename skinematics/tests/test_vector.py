import numpy as np
import sys
import os

from numpy import array, r_, vstack, abs, sin
from numpy.linalg import norm

import unittest

dir_name = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(dir_name, "..")))
import vector, quat

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.delta = 1e-5

    def test_target2orient(self):
        a = [1,1,0]
        b = [1., 0, 1]
        angle = 45
        
        result = vector.target2orient(np.array([a,b]), 'Helmholtz')
        correct = np.array([[ 0., angle, 0.],
                            [-angle, 0.,  0.]] )
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        result = vector.target2orient(np.array([a,b]), 'Fick')
        correct = np.array([[angle, 0., 0],
                            [0, -angle, 0.]] )
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        q_angle = quat.deg2quat(angle)
        result = vector.target2orient(np.array([a,b]))
        correct = np.array([[0, 0, q_angle],
                            [0, -q_angle, 0.]] )
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        result = vector.target2orient(a)
        correct = np.array([0, 0, q_angle])
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        
    def test_normalize(self):
        result = vector.normalize([3, 0, 0])
        correct = array([[ 1.,  0.,  0.]])
        error = norm(result-correct)
        self.assertAlmostEqual(error, 0)
        
        # Ensure that 'normalize' does not modify the input values
        data = 10 * np.ones(5)
        vector.normalize(data[:3])
        self.assertEqual(data[2], 10)
        
        
    def test_project(self):
        v1 = np.array([[1,2,3],
                       [4,5,6]])
        v2 = np.array([[1,0,0],
                       [0,1,0]])
        correct = array([[ 1.,  0.,  0.],
                       [ 0.,  5.,  0.]])
        result = vector.project(v1,v2)
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
        
        # Test default "projection_type"
        result = vector.project(v1,v2, projection_type='1D')
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
        
        # Test with lists
        v1 = list(v1[0])
        v2 = list(v2[0])
        correct = correct[0]
        result = vector.project(v1, v2)
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
        
    def test_angle(self):
        v1 = np.array([[1,0,0],
                       [1,1,0]])
        v2 = np.array([[1,0,0],
                       [2,0,0]])
        result = vector.angle(v1,v2)
        correct = r_[0, np.pi/4]
        self.assertAlmostEqual(np.linalg.norm(correct-result),0)
        self.assertAlmostEqual(vector.angle(v1[0], v1[1]), np.pi/4)
        
        result = vector.angle(v1,v2[0])
        correct = r_[0, np.pi/4]
        self.assertAlmostEqual(np.linalg.norm(correct-result),0)
        
        
    def test_GramSchmidt(self):
        P1 = np.array([[0, 0, 0], [1,2,3]])
        P2 = np.array([[1, 0, 0], [4,1,0]])
        P3 = np.array([[1, 1, 0], [9,-1,1]])
        result = vector.GramSchmidt(P1,P2,P3)
        correct = array([[ 1.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  1.        ],
       [ 0.6882472 , -0.22941573, -0.6882472 ,  0.62872867, -0.28470732,
         0.72363112, -0.36196138, -0.93075784, -0.05170877]])
        
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
    
    def test_GramSchmidt_vector(self):
        P1 = np.array([0, 0, 0])
        P2 = np.array([2, 0, 0])
        P3 = np.array([1, 1, 0])
        result = vector.GramSchmidt(P1,P2,P3)
        correct = array([ 1.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  1.        ])
        
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
    
    def test_planeOrientation(self):
        P1 = np.array([[0, 0, 0], [0,0,0]])
        P2 = np.array([[1, 0, 0], [0,1,0]])
        P3 = np.array([[1, 1, 0], [0,0,1]])
        
        result = vector.plane_orientation(P1,P2,P3)
        correct = array([[ 0.,  0., 1.],
                    [1.,  0.,  0.]])
        
        self.assertTrue(np.all(np.abs(result-correct)<self.delta))
        
    def test_qrotate(self):
        v1 = np.array([[1,0,0],
                       [1,1,0]])
        v2 = np.array([[1,1,0],
                       [2,0,0]])
        
        correct = array([[ 0.,0.,0.38268343],
                         [ 0.,0.,-0.38268343]])
        self.assertTrue(np.all(np.abs(vector.q_shortest_rotation(v1,v2)-correct)<self.delta))
        
        correct = array([ 0.,0.,0.38268343])
        self.assertTrue(norm(vector.q_shortest_rotation(v1[0],v2[0])-correct)<self.delta)
        
    def test_rotate_vector(self):
        x = [[1,0,0], [0, 1, 0], [0,0,1]]
        result = vector.rotate_vector(x, [0, 0, sin(0.1)])
        correct = array([[ 0.98006658,  0.19866933,  0.        ],
              [-0.19866933,  0.98006658,  0.        ],
              [ 0.        ,  0.        ,  1.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        

if __name__ == '__main__':
    #unittest.main()
    print('Thanks for using programs from Thomas!')    
