import sys
import os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import unittest
import numpy as np
from time import sleep
import view

class TestSequenceFunctions(unittest.TestCase):

    def test_view(self):
        t = np.arange(0,10,0.1)
        x = np.sin(t) + 0.2*np.random.randn(len(t))
        data = np.random.randn(100,3)
        view.ts(x)
        #viewer.ts(locals())
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
