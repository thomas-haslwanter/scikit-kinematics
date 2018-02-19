'''
Import data saved with "x-IMU" sensors from x-io, through subclassing "IMU_Base"
Note that the data are in two files:

    - a data-file
    - a reg-file

More info about the sensor on http://x-io.co.uk

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

class XIO(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """    
    
    def get_data(self, in_file, in_data=None):
        '''
        Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".

        Assigns the following properties
            - rate : rate
            - acc : acceleration
            - omega : angular_velocity
            - mag : mag_field_direction
            - packet_nr : packet_nr
        
        Parameters
        ----------
        in_selection : string
                Directory containing all the data-files, or
                filename of one file in that directory
        in_data : not used here
        
        '''
        in_selection = in_file
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
        
        # Set the class properties
        in_data = {'rate':rates['InertialAndMagnetic'],
               'acc':   data[0],
               'omega': data[1],
               'mag':   data[2]}
        self._set_data(in_data)
        
        self.packet_nr = data[-1]

if __name__ == '__main__':
    test_dir = r'../tests/data/data_xio'    
    assert os.path.exists(test_dir)
    
    my_sensor = XIO(in_file=test_dir)
    
    import matplotlib.pyplot as plt    
    
    plt.plot(my_sensor.acc)    
    print(my_sensor.rate)
    plt.show()
    print('Done')
    
