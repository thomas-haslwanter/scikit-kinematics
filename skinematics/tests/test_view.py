import sys
import os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
import numpy as np
from time import sleep
import view, quat

class TestSequenceFunctions(unittest.TestCase):

    def test_view_ts(self):
        t = np.arange(0,10,0.1)
        x = np.sin(t) + 0.2*np.random.randn(len(t))
        data = np.random.randn(100,3)
        view.ts(x)
        view.ts(locals())
        
    def test_view_orientation(self):
        omega = np.r_[0, 10, 10]     # [deg/s]
        duration = 2
        rate = 100
        q0 = [1, 0, 0, 0]
        out_file = 'demo_patch.mp4'
        title_text = 'Rotation Demo'
        
        ## Calculate the orientation
        dt = 1./rate
        num_rep = duration*rate
        omegas = np.tile(omega, [num_rep, 1])
        quaternion = quat.calc_quat(omegas, q0, rate, 'sf')
            
        view.orientation(quaternion, out_file, 'Well done!')        
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
