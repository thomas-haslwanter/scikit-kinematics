import sys
import os
sys.path.insert(0, os.path.abspath(r'..'))

import unittest
import numpy as np
from time import sleep
import viewer

class TestSequenceFunctions(unittest.TestCase):

    def test_viewer(self):
        t = np.arange(0,10,0.1)
        x = np.sin(t) + 0.2*np.random.randn(len(t))
        data = np.random.randn(100,3)
        viewer.ts(x)
        #viewer.ts(locals())
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
