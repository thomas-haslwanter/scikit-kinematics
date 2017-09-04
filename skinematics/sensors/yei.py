'''
Import data saved with yei-sensors, through subclassing "IMU_Base"
'''

'''
Author: Thomas Haslwanter
Version: 0.1
Date: Sept-2017
'''

import numpy as np
import pandas as pd
import re

import abc
import sys
sys.path.append("..")
from skinematics.imus import IMU_Base

class YEI(IMU_Base):
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
        
        data = pd.read_csv(in_file)
        
        # Generate a simple list of column names
        newColumns = []
        pattern = re.compile('.*%(\w+)\((\w+)\)')
        for name in data.columns:
            newColumns.append(pattern.match(name).groups()[1])
        data.columns = newColumns
        
        
        # Calculate rate (ChipTime is in microsec)
        start = data.ChipTimeUS[0] * 1e-6    # microseconds to seconds
        stop = data.ChipTimeUS.values[-1] * 1e-6    # pandas can't count backwards
        rate = len(data) / (stop-start)
        
        returnValues = [rate]
        
        # Extract the columns that you want, by name
        paramList=['Accel', 'Gyro', 'Compass']
        for Expression in paramList:
            returnValues.append(data.filter(regex=Expression).values)
        
        self._set_info(*returnValues)

if __name__ == '__main__':
    my_sensor = YEI(in_file=r'..\tests\data\data_yei.txt')    
    
    import matplotlib.pyplot as plt    
    
    plt.plot(my_sensor.acc)    
    plt.show()
    print('Done')
    