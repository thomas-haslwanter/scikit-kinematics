"""
This file can be used to test quaterion viewers.

author: thomas haslwanter
date:   Jan-2018
ver:    0.1
"""
import numpy as np
import skinematics as skin

def make_positive_rotations():
    '''Generate a quaternion that rotates to the left, down, and CW'''
    
    
    phi = np.zeros(1000)
    phi[200:500] = np.linspace(0, 90, 300)
    phi[500:800] = np.linspace(90, 0, 300)
    quat = skin.quat.deg2quat(phi)
    
    q_out = np.zeros( (3000, 3) )
    q_out[:1000,2] = quat
    q_out[1000:2000,1] = quat
    q_out[-1000:,0] = quat
    
    return q_out
