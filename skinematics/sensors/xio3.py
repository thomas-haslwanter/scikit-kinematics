'''
Import data saved with "x-IMU3" sensors from X-IO, through subclassing "IMU_Base"
Note that the data are in three files: data come from

    - an inertial sensor
    - a magnetometer
    - a high-speed accelerometer

More info about the sensor on https://x-io.co.uk/x-imu3/
'''

# Author: Thomas Haslwanter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants
import os

from skinematics.imus import IMU_Base


def read_datafiles(in_files, rate=50):
    '''Read data from XIO3-files.
    The data from inertial- and magnet-sensor sensors are sampled at
    different frequencies. To nevertheless provide a fixed frequency, the data
    are re-sampled to "rate".

    Parameters
    ----------

    in_file : list with file-paths of the data-files
    rate : sample-rate for the interpolated data (Hz)

    Returns
    -------
    out_list: list
            Contains the following parameters:

            - time [s]
            - acceleration [g]
            - angular_velocity [deg/s]
            - mag_field_direction [uT]
    '''

    data_imu = pd.read_csv(in_files['IMU']).values
    data_b = pd.read_csv(in_files['mag']).values
    # df_acc = pd.read_csv(in_files['HGacc'])

    t_i = data_imu[:, 0]
    t_b = data_b[:, 0]
    # t_acc = df_acc['Timestamp (us)']

    start = np.max([t_i[0], t_b[0]])
    stop = np.min([t_i[-1], t_b[-1]])
    step = 1e6 / rate   # since the timestamps are in micro-seconds

    # Interpolate the IMU-data
    t = np.arange(start, stop, step)
    data = []
    for col in range(1, 7):
        data.append(np.interp(t, t_i, data_imu[:, col]))

    # Interpolate the magnetic field measurements
    for col in range(1, 4):
        data.append(np.interp(t, t_b, data_b[:, col]))

    data = np.array(data).T
    acc = data[:, 3:6]
    omega = data[:, :3]
    b = data[:, 6:]

    return (acc, omega, b)


class XIO3(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """

    def get_data(self, in_file, in_data):

        '''
        Get the recorded data, and assign them to the corresponding attributes
        of "self". For the XIO3, the data are stored in different files:

        - "Inertial.csv" : data from the inertial sensor
        - "Magnetometer.csv" : data from the magnetometer
        - "HighGAccelerometer.csv" : data from the high-speed-accelerometer

        For the protocoll here, data from the HighGAccelerometer are ignored,
        data from the magnetometer are interpolated to provide the same frequency
        as the data from the inertial sensor.

        Assigns the following properties:
            - time : recording, starting at 0 (sec)
            - acc : acceleration (g)
            - omega : angular_velocity (deg/s)
            - mag : mag_field_direction ()

        Parameters
        ----------
        in_selection : string
                Directory containing all the data-files, or
                filename of one file in that directory
        in_data : requested sample rate for interpolation


        '''

        in_selection = in_file
        rate = in_data

        if os.path.isdir(in_selection):
            in_dir = in_selection
        else:
            in_file = in_selection
            in_dir = os.path.split(in_file)[0]

        file_list = os.listdir(in_dir)

        # Get the filenames, based on the XIO3-definitions
        files = {}
        for file in file_list:
            if file.find('Inertial') >= 0:
                files['IMU'] = os.path.join(in_dir, file)
            if file.find('Magnetometer') >= 0:
                files['mag'] = os.path.join(in_dir, file)
            if file.find('HighGAccelerometer') >= 0:
                files['HGacc'] = os.path.join(in_dir, file)

        # Read the sensor-data
        acc, omega, b = read_datafiles(files, rate)

        # Set the class properties
        in_data = {'rate': rate,
               'acc':   acc * constants.g,
               'omega': np.deg2rad(omega),
               'mag':   b}
        self._set_data(in_data)


if __name__ == '__main__':
    test_dir = r'..\..\tests\data\data_xio3'
    assert os.path.exists(test_dir)

    sample_rate = 50. # Hz
    my_sensor = XIO3(in_file=test_dir, in_data=sample_rate)

    plt.plot(my_sensor.acc)
    print(my_sensor.rate)
    plt.show()
    print('Done')

