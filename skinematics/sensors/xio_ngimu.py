'''
Import data saved with "NGIMU" sensors from x-io, through subclassing "IMU_Base"
Note that the data are in two files:

    - a data-file
    - a reg-file

More info about the sensor on http://x-io.co.uk/ngimu

'''

# Author: Thomas Haslwanter

import numpy as np
import pandas as pd
import os
from scipy import constants
import abc

from skinematics.imus import IMU_Base

def read_ratefile(reg_file):
    '''Read send-rates from an NGIMU sensor.
    "Disabled" channels have the "rate" set to "None".

    Parameters
    ----------
    in_file : string
            Has to be the "Registers"-file.

    Returns
    -------
    rates: directory
            Contains the send-rates for the following paramters:
            - sensors
            - magnitudes
            - quaternion
            - matrix
            - euler
            - linear
            - earth
            - altitude
            - temperature
            - humidity
            - battery
            - analogue
            - rssi

    '''

    rates = {}

    # The the file content
    with open(reg_file, 'r') as in_file:
        lines = in_file.readlines()

    # Get the send rates
    for line in lines:
        if line.find('rate') > 0:
            # e.g.: "/rate/quaternion, 50f"
            param_full, val_txt = line.split()
            value = np.float64(val_txt[:-1])  # skip the "f"
            param = param_full.split('/')[2][:-1]
            if value > 0:
                rates[param] = value
            else:
                rates[param] = None

    return rates

def read_datafile(in_file):
    '''Read data from an XIO "CalInertialAndMag"-file.

    Parameters
    ----------
    in_file : string
            Has to be the name of the "sensors.csv"-file.

    Returns
    -------
    out_list: list
            Contains the following parameters:

            - time [s]
            - acceleration [g]
            - angular_velocity [deg/s]
            - mag_field_direction [uT]
            - barometer [hPa]
    '''

    data = pd.read_csv(in_file)
    out_list = []

    # Extract the columns that you want, by name
    param_list=['Time', 'Acc', 'Gyr', 'Mag', 'Baro']
    for Expression in param_list:
        out_list.append(data.filter(regex=Expression).values)

    return out_list

class NGIMU(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """

    def get_data(self, in_file, in_data=None):
        '''
        Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".
        For the x-io.NGIMU, the data are stored in different files:

        - "Settings.txt" : contains all the sensor-settings
        - "sensors.csv" : contains the recorded sensor data

        Assigns the following properties:
            - rate : rate
            - acc : acceleration
            - omega : angular_velocity
            - mag : mag_field_direction

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
            if file.find('Settings') >= 0:
                files['settings'] = os.path.join(in_dir, file)
            if file.find('sensors') >= 0:
                files['data'] = os.path.join(in_dir, file)

        # Read in the registers-file, and extract the sampling rates
        rates = read_ratefile(files['settings'])

        # Read the sensor-data
        data  = read_datafile(files['data'])

        # Set the class properties
        in_data = {'rate':rates['sensors'],
               'acc':   data[1] * constants.g,
               'omega': np.deg2rad(data[2]),
               'mag':   data[3]}
        self._set_data(in_data)


if __name__ == '__main__':
    test_dir = r'../../tests/data/data_ngimu'
    assert os.path.exists(test_dir)

    my_sensor = NGIMU(in_file=test_dir)

    import matplotlib.pyplot as plt

    plt.plot(my_sensor.acc)
    print(my_sensor.rate)
    plt.show()
    print('Done')

