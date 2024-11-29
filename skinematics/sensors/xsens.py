'''
Import data saved with XSens-sensors, through subclassing "IMU_Base"
'''

# Author: Thomas Haslwanter

import numpy as np
import pandas as pd
import abc

# To ensure that the relative path works
import os
import sys

from skinematics.imus import IMU_Base

class XSens(IMU_Base):
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

        # Get the sampling rate from the second line in the file
        try:
            fh = open(in_file)
            fh.readline()
            line = fh.readline()
            rate = np.float64(line.split(':')[1].split('H')[0])
            fh.close()

        except FileNotFoundError:
            print('{0} does not exist!'.format(in_file))
            return -1

        # Read the data
        data = pd.read_csv(in_file,
                           sep='\t',
                           skiprows=4,
                           index_col=False)

        # Extract the columns that you want, and pass them on
        in_data = {'rate':rate,
               'acc':   data.filter(regex='Acc').values,
               'omega': data.filter(regex='Gyr').values,
               'mag':   data.filter(regex='Mag').values}
        self._set_data(in_data)

if __name__ == '__main__':
    my_sensor = XSens(in_file=r'..\..\tests\data\data_xsens.txt')

    import matplotlib.pyplot as plt

    plt.plot(my_sensor.quat[:,1:])
    plt.show()
    print('Done')

