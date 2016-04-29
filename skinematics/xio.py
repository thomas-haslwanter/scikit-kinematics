'''
Handle data saved with XIO-sensors
'''

'''
Author: Thomas Haslwanter
Version: 0.1
Date: April-2016
'''

import os
import pandas as pd

def find_rates(reg_file):
    '''Read send-rates from an XIO sensor.
    "Disabled" channels have the "rate" set to "None".
    
    Inputs:
    -------
    in_file : string
            Has to be the "Registers"-file
    
    Returns:
    --------
    rates: directory
            Contains the send-rates for the different "params"
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
                    
def read_data(in_file):
    '''Read data from an XIO sensor.
    
    Inputs:
    -------
    in_file : string
            Has to be the "CalInertialAndMag"-file
    
    Returns:
    --------
    out_list: list
            Contains the following parameters:
            acceleration
            angular_velocity
            mag_field_direction
            packet_nr
    '''
    
    data = pd.read_csv(in_file)
    out_list = []
    
    # Extract the columns that you want, by name
    param_list=['Acc', 'Gyr', 'Mag', 'Packet']
    for Expression in param_list:
        out_list.append(data.filter(regex=Expression).values)
        
    return out_list

def get_infos(in_dir):
    '''Get the sampling rates, as well as the recorded data.
    
    Inputs:
    -------
    in_dir : string
            Directory containing all the data-files
    
    Returns:
    --------
    out_list: list
            Contains the following parameters:
            rate
            acceleration
            angular_velocity
            mag_field_direction
            packet_nr
    '''
    
    file_list = os.listdir(in_dir)
   
    # XIO-definition of the sampling rate is in the "Registers"-file
    files = {}
    for file in file_list:
        if file.find('Registers') > 0:
            files['register'] = os.path.join(in_dir, file)
        if file.find('CalInertialAndMag') > 0:
            files['data'] = os.path.join(in_dir, file)
    
    # Read in the registers-file, and extract the sampling rates
    rates = find_rates(files['register'])
    data = read_data(files['data'])
    
    return ([rates['InertialAndMagnetic']] + data)
            
if __name__=='__main__':
    test_dir = r'.\tests\data\data_xio'    
    assert os.path.exists(test_dir)
    
    data = get_infos(test_dir)
    
    print('Rate: {0}'.format(data[0]))
    print('Acceleration: {0}'.format(data[1]))
    
    