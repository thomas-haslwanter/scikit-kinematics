'''
Import data saved with XSens-sensors
'''

'''
Author: Thomas Haslwanter
Version: 0.1
Date: May-2016
'''

import numpy as np
import pandas as pd

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
    
    # Get the sampling rate from the second line in the file
    try:
        fh = open(in_file)
        fh.readline()
        line = fh.readline()
        rate = np.float(line.split(':')[1].split('H')[0])
        fh.close()
        returnValues = [rate]

    except FileNotFoundError:
        print('{0} does not exist!'.format(in_file))
        return -1

    # Read the data
    data = pd.read_csv(in_file,
                       sep='\t',
                       skiprows=4, 
                       index_col=False)

    # Extract the columns that you want, by name
    paramList=['Acc', 'Gyr', 'Mag']
    for param in paramList:
        Expression = param + '*'
        returnValues.append(data.filter(regex=Expression).values)

    return returnValues

