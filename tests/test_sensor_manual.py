''' Test manual data entry, through subclassing "IMU_Base" '''

# Author: Thomas Haslwanter

import unittest
from time import sleep
import os
from skinematics.sensors.xsens import XSens
from skinematics.sensors.manual import MyOwnSensor

class TestSequenceFunctions(unittest.TestCase):

    def test_import_manual(self):
        # Get data, with a specified input from an XSens system
        in_file = os.path.join('.', 'data', 'data_xsens.txt')

        sensor = XSens(in_file=in_file, q_type=None)

        transfer_data = {'rate':sensor.rate,
                   'acc': sensor.acc,
                   'omega':sensor.omega,
                   'mag':sensor.mag}
        my_sensor = MyOwnSensor(in_file='My own 123 sensor.', in_data=transfer_data)

        self.assertEqual(my_sensor.rate, 50.)
        self.assertAlmostEqual( (my_sensor.omega[0,2] - 0.050860000000000002), 0)


    #def test_pos(self):
        ## suggested in github


if __name__ == '__main__':
    unittest.main()
    print('Thanks for using programs from Thomas!')
    sleep(0.1)
