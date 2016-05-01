'''
Import data saved with XIO-sensors
'''

'''
Author: Thomas Haslwanter
Version: 0.2
Date: May-2016
'''

import os
import pandas as pd

def read_ratefile(reg_file):
    '''Read send-rates from an XIO sensor.
    "Disabled" channels have the "rate" set to "None".
    
    Parameters
    ----------
    in_file : string
            Has to be the "Registers"-file.
    
    Returns
    -------
    rates: directory
            Contains the send-rates for the different "params".
    '''
    
    params = ['Sensor', 
              'DateTime',
              'BatteryAndThermometer',
              'InertialAndMagnetic',
              'Quaternion'
            ]
    
    rates = {}
    
    # The the file content
    with open(reg_file, 'r') as in_file:
        lines = in_file.readlines()
    
    # Get the send rates
    for param in params:
        for line in lines:
            if line.find(param) > 0:
                rate_flag = int(line.split(',')[2])
                if rate_flag:
                    '''
                    0 ... 1 Hz
                    1 ... 2 Hz
                    10 ... 512 Hz
                    '''
                    rates[param] = 2 ** (rate_flag-1)
                else:
                    # Disabled
                    rates[param] = None
    
    return rates
                    
def read_datafile(in_file):
    '''Read data from an XIO "CalInertialAndMag"-file.
    
    Parameters
    ----------
    in_file : string
            Has to be the name of the "CalInertialAndMag"-file.
    
    Returns
    -------
    out_list: list
            Contains the following parameters:
            
            - acceleration
            - angular_velocity
            - mag_field_direction
            - packet_nr
    '''
    
    data = pd.read_csv(in_file)
    out_list = []
    
    # Extract the columns that you want, by name
    param_list=['Acc', 'Gyr', 'Mag', 'Packet']
    for Expression in param_list:
        out_list.append(data.filter(regex=Expression).values)
        
    return out_list

def get_data(in_selection):
    '''Get the sampling rates, as well as the recorded data.
    
    Parameters
    ----------
    in_selection : string
            Directory containing all the data-files, or
            filename of one file in that directory
    
    Returns
    -------
    out_list: list
            Contains the following parameters:
            
            - rate
            - acceleration
            - angular_velocity
            - mag_field_direction
            - packet_nr
    '''
    
    if os.path.isdir(in_selection):
        in_dir = in_selection
    else:
        in_file = in_selection
        in_dir = os.path.split(in_file)[0]
        
    file_list = os.listdir(in_dir)
   
    # Get the filenames, based on the XIO-definitions
    files = {}
    for file in file_list:
        if file.find('Registers') > 0:
            files['register'] = os.path.join(in_dir, file)
        if file.find('CalInertialAndMag') > 0:
            files['data'] = os.path.join(in_dir, file)
    
    # Read in the registers-file, and extract the sampling rates
    rates = read_ratefile(files['register'])
    
    # Read the sensor-data
    data  = read_datafile(files['data'])
    
    return ([rates['InertialAndMagnetic']] + data)
            
if __name__=='__main__':
    test_dir = r'..\..\tests\data\data_xio'    
    assert os.path.exists(test_dir)
    
    data = get_data(test_dir)
    
    print('Rate: {0} [Hz]'.format(data[0]))
    print('Acceleration [m/s^2]:\n {0}'.format(data[1]))
    
    
