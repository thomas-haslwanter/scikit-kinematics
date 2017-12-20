'''
Import data manually, through subclassing "IMU_Base"
'''

'''
Author: Thomas Haslwanter
'''

import numpy as np
import pandas as pd
import abc

# To ensure that the relative path works
import os
import sys
dir_name = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(dir_name, "..")))

from imus import IMU_Base

class MyOwnSensor(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """    
    
    def get_data(self, in_file=None, in_data=None):
        '''Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".
        
        Parameters
        ----------
        in_file : string
                Information about the data origin
        in_data : MUST contain rate / acc / omega, mag is optional
        
        Assigns
        -------
        - rate : rate
        - acc : acceleration
        - omega : angular_velocity
        - mag : mag_field_direction
        '''
        
        # Get the sampling rate from the second line in the file
        transfer_data = {}
        try:
            paramList=['rate', 'acc', 'omega']
            for param in paramList:
                transfer_data[param] = in_data[param]
        except KeyError:
            print('{0} is a required argument for manual data input!', param)
            raise
    
        if 'mag' in in_data.keys():
            transfer_data['mag'] = in_data['mag']
                          
        self._set_data(transfer_data)

if __name__ == '__main__':
    from skinematics.sensors.xsens import XSens
    import matplotlib.pyplot as plt    
    
    xsens_sensor = XSens(in_file=r'..\tests\data\data_xsens.txt')    
    #xsens_sensor = XSens(in_file=r'..\tests\data\data_xsens.txt', q_type=None)    
    
    in_data = {'rate':xsens_sensor.rate,
               'acc': xsens_sensor.acc,
               'omega':xsens_sensor.omega,
               'mag':xsens_sensor.mag}
    my_sensor = MyOwnSensor(in_file='My own 123 sensor.', in_data=in_data)
    print(my_sensor.omega[:3,:])
    
    plt.plot(my_sensor.acc)    
    plt.show()
    print('Done')
    