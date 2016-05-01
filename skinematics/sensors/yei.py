'''
Import data saved with YEI-sensors
'''

'''
Author: Thomas Haslwanter
Version: 0.1
Date: May-2016
'''

import pandas as pd
import re

def get_data(in_file):
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
        
    return returnValues

