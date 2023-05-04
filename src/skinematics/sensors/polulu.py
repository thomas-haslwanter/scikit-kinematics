'''
Import data saved with polulu-sensors, through subclassing "IMU_Base"

https://www.pololu.com/product/2738
These are low-cost IMUS (<20 US$), where acceleration/gyroscope data are not
sampled at the same time as the magnetic field data (just over 100 Hz).
As a result, the interpolated sampling rate has to be set by hand.
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

parent_dir = os.path.abspath(os.path.join( os.path.dirname(__file__), '..' ))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from imus import IMU_Base

class Polulu(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """    
    
    def get_data(self, in_file, in_data=125):
        '''Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".
        
        Parameters
        ----------
        in_file : string
                Filename of the data-file
        in_data : float
                Sampling rate (has to be provided!!)
        
        Assigns
        -------
        - rate : rate
        - acc : acceleration
        - omega : angular_velocity
        - mag : mag_field_direction
        '''
        
        try:
            # The sampling rate has to be provided externally
            rate = in_data['rate']
            
            # Get the data, and label them
            data = pd.read_csv(in_file, header=None, delim_whitespace=True, engine='python')
            data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'taccgyr', 'tmag']
            
            # interpolate with a manually set rate. Note that this sensor acquires exactly 25 seconds!
            dt = 1/float(rate)
            t_lin = np.arange(0, 25, dt)
    
            data_interp = pd.DataFrame()
            # Different sampling times for acc/gyr and for mag!
            for ii in range(6):
                data_interp[data.keys()[ii]] = np.interp(t_lin*1000, data['taccgyr'], data.iloc[:,ii])
            for ii in range(6,9):
                data_interp[data.keys()[ii]] = np.interp(t_lin*1000, data['tmag'], data.iloc[:,ii])
            data_interp['time'] = t_lin
            
            # Set the conversion factors by hand, and apply them
            conversions = {}
            conversions['mag'] = 1/6842
            conversions['acc'] = 0.061/1000
            conversions['gyr'] = 4.375/1000 * np.pi/180
            
            data_interp.iloc[:,:3] *= conversions['acc']
            data_interp.iloc[:,3:6] *= conversions['gyr']
            data_interp.iloc[:,6:9] *= conversions['mag']
            
        except FileNotFoundError:
            print('{0} does not exist!'.format(in_file))
            return -1
    
        # Extract the columns that you want, and pass them on
        in_data = {'rate':rate,
               'acc':   data_interp.filter(regex='acc').values,
               'omega': data_interp.filter(regex='gyr').values,
               'mag':   data_interp.filter(regex='mag').values}
        self._set_data(in_data)

if __name__ == '__main__':
    inFile = r'..\tests\data\data_polulu.txt'
    in_data = {'rate':125}
    my_sensor = Polulu(in_file=inFile, in_data=in_data)
    
    import matplotlib.pyplot as plt    
    
    plt.plot(my_sensor.acc)    
    plt.show()
    print(my_sensor.rate)
    print('Done')
    