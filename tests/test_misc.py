import unittest
from time import sleep
from skinematics import misc

class TestSequenceFunctions(unittest.TestCase):

    def test_progressbar(self):
        '''Just run it through once'''
        for ii in misc.progressbar(range(50), 'Computing ', 25):
            #print(ii)
            sleep(0.05)
        
    def test_get_screensize(self):
        width, height = misc.get_screensize()
        print('Your screen is {0}x{1}'.format(width, height))
        
if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(2)
