import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))

import unittest
import numpy as np
from numpy import sin, cos, array, r_, vstack, abs, tile
from numpy.linalg import norm
import quat
from time import sleep

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.qz  = r_[cos(0.1), 0,0,sin(0.1)]
        self.qy  = r_[cos(0.1),0,sin(0.1), 0]

        self.quatMat = vstack((self.qz,self.qy))

        self.q3x = r_[sin(0.1), 0, 0]
        self.q3y = r_[2, 0, sin(0.1), 0]

        self.delta = 1e-4

    def test_deg2quat(self):
        self.assertAlmostEqual(quat.deg2quat(10), 0.087155742747658166)
        
        result = quat.deg2quat(array([10, 170, 190, -10]))
        correct =  array([ 0.08715574,  0.9961947 , -0.9961947 , -0.08715574])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
 
    def test_quat2rotmat(self):
        result = quat.quat2rotmat(array([0, 0, 0.1]))
        correct = array([[ 0.98      , -0.19899749,  0.        ],
                   [ 0.19899749,  0.98      ,  0.        ],
                   [ 0.        ,  0.        ,  1.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quat2rotmat([[0, 0, 0.1], [0, 0.1, 0]])
        correct = array([[ 0.98      , -0.19899749,  0.        ,  0.19899749,  0.98      ,
                 0.        ,  0.        ,  0.        ,  1.        ],
               [ 0.98      ,  0.        ,  0.19899749,  0.        ,  1.        ,
                 0.        , -0.19899749,  0.        ,  0.98      ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        
    def test_quat2vect(self):
        result = quat.quat2vect([cos(0.1), 0, 0, sin(0.1)])
        correct = array([ 0.        ,  0.        ,  0.09983342])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        
        result = quat.quat2vect([[cos(0.1), 0., 0, sin(0.1)],
             [cos(0.2), 0, sin(0.2), 0]])
        correct = array([[ 0.        ,  0.        ,  0.09983342],
               [ 0.        ,  0.19866933,  0.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        
    def test_quatmult(self):
        result = quat.quatmult(self.qz, self.qz) 
        correct =  array([ 0.98006658,  0.        ,  0.        ,  0.19866933])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.qz, self.qy) 
        correct = array([ 0.99003329, -0.00996671,  0.09933467,  0.09933467])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.quatMat, self.quatMat) 
        correct = array([[ 0.98006658,  0.        ,  0.        ,  0.19866933],
       [ 0.98006658,  0.        ,  0.19866933,  0.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.quatMat, self.qz) 
        correct = array([[ 0.98006658,  0.        ,  0.        ,  0.19866933],
       [ 0.99003329,  0.00996671,  0.09933467,  0.09933467]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.q3x, self.q3x) 
        correct = array([ 0.19866933,  0.        ,  0.        ])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

    def test_quatconj(self):
        q= [0, 0, 1]
        result = quat.quatconj(q)
        correct =  array([ 0.,  0.,  0.,  -1.])
        error = norm(result - correct)
        self.assertAlmostEqual(error, 0)

        q = array([[cos(0.1),0,0,sin(0.1)],
            [cos(0.2), 0, sin(0.2), 0]])
        result = quat.quatconj(q)
        correct =  array([[ 0.99500417, -0.        , -0.        , -0.09983342],
       [ 0.98006658, -0.        , -0.19866933, -0.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

    def test_quatinv(self):
        result = quat.quatmult(self.qz, quat.quatinv(self.qz)) 
        correct =  array([ 1.,  0.,  0.,  0.])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.quatMat, quat.quatinv(self.quatMat)) 
        correct =  array([[ 1.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quatmult(self.q3x, quat.quatinv(self.q3x)) 
        correct =  array([ 0.,  0.,  0.])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)


    def test_quat2deg(self):
        result = quat.quat2deg(self.qz)
        correct =  array([  0.       ,   0.       ,  11.4591559])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quat2deg(self.quatMat)
        correct =  array([[  0.       ,   0.       ,  11.4591559],
       [  0.       ,  11.4591559,   0.       ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)

        result = quat.quat2deg(0.2)
        correct =  array([ 23.07391807])
        error = abs(result - correct)
        self.assertTrue(error < self.delta)

    def test_rotmat2quat(self):
        result = quat.rotmat2quat(quat.quat2rotmat([0, 0, 0.1]))
        correct = array([[ 0.99498744,  0.        ,  0.        ,  0.1       ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
    
    def test_vect2quat(self):
        result = quat.vect2quat([[0,0,sin(0.1)],[0,sin(0.2),0]])
        correct = array([[ 0.99500417,  0.        ,  0.        ,  0.09983342],
               [ 0.98006658,  0.        ,  0.19866933,  0.        ]])
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        
    def test_Quaternions(self):
        q = quat.Quaternion(np.array([0,0,0.5]))
        p = quat.Quaternion(np.array([[0,0,0.5], [0,0,0.1]]))
        print(p*q)
        print(q*3)
        print(q*np.pi)
        print(q/p)
        q5 = q/5
        error = q5.values[0,3]-0.1
        self.assertTrue(error < self.delta)
        self.assertEqual(p[0].values[0,3], 0.5)

    def test_vel2quat(self):
        v0 = [0., 0., 100.]
        vel = tile(v0, (1000,1))
        rate = 100
        out = quat.vel2quat(np.deg2rad(vel), [0., 0., 0.], rate, 'sf')
        result = out[-1:]
        correct =  [[-0.76040597, 0., 0., 0.64944805]]
        error = norm(result - correct)
        self.assertTrue(error < self.delta)
        
    def test_quat2vel(self):
        rate = 1000
        t = np.arange(0,10,1./rate)
        x = 0.1 * np.sin(t)
        y = 0.2 * np.sin(t)
        z = np.zeros_like(t)
        
        q = np.column_stack( (x,y,z) )
        vel = quat.quat2vel(q, rate, 5, 2)
        qReturn = quat.vel2quat(vel, q[0], rate, 'sf' )
        error = np.max(np.abs( q-qReturn[:,1:] ))
        self.assertTrue(error < 1e3 )
    
if __name__ == '__main__':
    q = quat.Quaternion(np.array([0,0,0.5]))
    p = quat.Quaternion(np.array([[0,0,0.5], [0,0,0.1]]))
    print(p*q)
    print(q*3)
    print(q*np.pi)
    print(q/p)
    #unittest.main()
    #print('Thanks for using programs from Thomas!')
    #sleep(2)
