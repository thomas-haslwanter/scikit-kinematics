'''
Import data saved with polulu-sensors
https://www.pololu.com/product/2738
These are low-cost IMUS (<20 US$), where acceleration/gyroscope data are not
sampled at the same time as the magnetic field data (just over 100 Hz).
As a result, the interpolated sampling rate has to be set by hand.
'''

'''
Author: Thomas Haslwanter
Version: 0.2
Date: May-2016
'''

import numpy as np
import pandas as pd

class FileNotFoundError(OSError):
    pass

def get_data(inFile, rate=125):
    '''Get the sampling rate, as well as the recorded data.
    
    Parameters
    ----------
    in_file : string
            Filename of the data-file
    
    Returns
    -------
    out_list: list
            Contains the following parameters:
            
            - rate
            - acceleration
            - angular_velocity
            - mag_field_direction
    '''
    
    # Get the sampling rate from the second line in the file
    try:
        # Get the data, and label them
        data = pd.read_csv(inFile, header=None, sep='[ ]*', engine='python')
        data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'taccgyr', 'tmag']
        
        # interpolate with a manually set rate
        dt = 1/np.float(rate)
        t_lin = np.arange(0, 25, dt)

        data_interp = pd.DataFrame()
        for ii in range(6):
            data_interp[data.keys()[ii]] = np.interp(t_lin*1000, data['taccgyr'], data.ix[:,ii])
        for ii in range(6,9):
            data_interp[data.keys()[ii]] = np.interp(t_lin*1000, data['tmag'], data.ix[:,ii])
        data_interp['time'] = t_lin
        
        # Set the conversion factors by hand, and apply them
        conversions = {}
        conversions['mag'] = 1/6842
        conversions['acc'] = 0.061/1000
        conversions['gyr'] = 4.375/1000 * np.pi/180
        
        data_interp.ix[:,:3] *= conversions['acc']
        data_interp.ix[:,3:6] *= conversions['gyr']
        data_interp.ix[:,6:9] *= conversions['mag']
        
    except FileNotFoundError:
        print('{0} does not exist!'.format(in_file))
        return -1

    returnValues = [rate]
    
    # Extract the columns that you want, by name
    paramList=['acc', 'gyr', 'mag']
    for param in paramList:
        Expression = param + '*'
        returnValues.append(data_interp.filter(regex=Expression).values)

    return returnValues

if __name__ == '__main__':
    inFile = r'..\tests\data\data_barnobi.txt'
    data = get_data(inFile)
    print(data[0])
    print(data[1])
    input('Done')
    
