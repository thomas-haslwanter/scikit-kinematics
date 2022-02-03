'''
Import data saved with Vicon-sensors, through subclassing "IMU_Base"
'''

'''
Author: Thomas Haslwanter
'''

import numpy as np
import pandas as pd
import abc

import btk
import scipy
import scipy.signal
import scipy.linalg

# To ensure that the relative path works
import os
import sys

parent_dir = os.path.abspath(os.path.join( os.path.dirname(__file__), '..' ))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from imus import IMU_Base

class Vicon(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """    
    
    def get_data(self, in_file, in_data=None):
        '''Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".
        
        Parameters
        ----------
        in_file : string
                Filename of the data-file
        in_data : not used here
        
        Assigns
        -------
        - rate : rate
        - acc : acceleration
        - omega : angular_velocity
        - mag : mag_field_direction
        '''        

        # Read the data
        # Get the sampling rate
        try:
            reader=btk.btkAcquisitionFileReader()
            reader.SetFilename(in_file)
            reader.Update()
            acq=reader.GetOutput()
            fp=acq.GetPointFrequency()
        except FileNotFoundError:
            print('{0} does not exist!'.format(in_file))
            return -1
    
        # Extract the columns that you want, and pass them on
        acc_x_values = acq.GetAnalog("accel.x").GetValues()
        acc_y_values = acq.GetAnalog("accel.y").GetValues()
        acc_z_values = acq.GetAnalog("accel.z").GetValues()

        gyro_x_values = acq.GetAnalog("gyro.x").GetValues()
        gyro_y_values = acq.GetAnalog("gyro.y").GetValues()
        gyro_z_values = acq.GetAnalog("gyro.z").GetValues()

        mag_x_values = acq.GetAnalog("mag.x").GetValues()
        mag_y_values = acq.GetAnalog("mag.y").GetValues()
        mag_z_values = acq.GetAnalog("mag.z").GetValues()


        in_data = {'rate':fp,
               'acc': np.column_stack((acc_x_values,acc_y_values,acc_z_values)),
               'omega': np.column_stack((gyro_x_values,gyro_y_values,gyro_z_values)),
               'mag':   np.column_stack((mag_x_values,mag_y_values,mag_z_values))}
        self._set_data(in_data)

if __name__ == '__main__':
    my_sensor = Vicon(in_file=r'..\tests\data\LeftFoot-marche01.c3d')    
    
    import matplotlib.pyplot as plt    
    
    plt.plot(my_sensor.quat[:,1:])    
    plt.show()
    print('Done')
    