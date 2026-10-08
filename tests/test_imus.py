# from .context import skinematics

import unittest
import numpy as np
from numpy import sin, cos, array, r_, vstack, abs, tile, pi
from numpy.linalg import norm
import os
from skinematics import imus, quat, vector, rotmat
from time import sleep
from skinematics.simulations.simulate_movements import simulate_imu

myPath = os.path.dirname(os.path.abspath(__file__))

# The VQF filter is an optional dependency: "pip install scikit-kinematics[vqf]"
try:
    import vqf

    vqf_installed = True
except ModuleNotFoundError:
    vqf_installed = False

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):

        # Those are currently not needed
        # self.qz  = r_[cos(0.1), 0,   0,    sin(0.1)]
        # self.qy  = r_[cos(0.1), 0, sin(0.1), 0]
        # self.quatMat = vstack((self.qz,self.qy))
        # self.q3x = r_[sin(0.1),  0,     0]
        # self.q3y = r_[  0,   sin(0.1), 0]
        # self.delta = 1e-4

        # Simulate IMU-data
        duration_movement = 1  # [sec]
        duration_total = 1  # [sec]
        rate = 100  # [Hz]

        B0 = vector.normalize([1, 0, 1])  # geomagnetic field, re earth

        rotation_axis = [0, 1, 0]
        angle = 90

        translation = [1, 0, 0]
        distance = 0

        self.q_init = [0, 0, 0]
        self.pos_init = [0, 0, 0]

        self.imu_signals, self.body_pos_orient = simulate_imu(
            rate,
            duration_movement,
            duration_total,
            q_init=self.q_init,
            rotation_axis=rotation_axis,
            deg=angle,
            pos_init=self.pos_init,
            direction=translation,
            distance=distance,
            B0=B0,
        )

    def test_analytical(self):

        # Analyze the simulated data with "analytical"
        imu = self.imu_signals

        q, pos, vel = imus.analytical(
            R_initialOrientation=np.eye(3),
            omega=imu["omega"],
            initialPosition=np.zeros(3),
            accMeasured=imu["gia"],
            rate=imu["rate"],
        )

        # and then check, if the position is [0,0,0], and the orientation-quat = [0, sin(45), 0]
        self.assertTrue(np.max(np.abs(pos[-1])) < 0.001)  # less than 1mm

        result = quat.q_vector(q[-1])
        correct = array([0.0, np.sin(np.deg2rad(45)), 0.0])
        error = norm(result - correct)
        self.assertAlmostEqual(error, 0)

    def test_kalman(self):

        # Analyze the simulated data with "kalman"
        imu = self.imu_signals
        q_kalman = imus.kalman(imu["rate"], imu["gia"], imu["omega"], imu["magnetic"])

        # and then check, if the quat_vector = [0, sin(45), 0]
        result = quat.q_vector(q_kalman[-1])
        correct = array([0.0, np.sin(np.deg2rad(45)), 0.0])  # [0, 0.71, 0]
        error = norm(result - correct)
        self.assertAlmostEqual(error, 0, places=2)

        # Get data
        inFile = os.path.join(myPath, "data", "data_xsens.txt")
        from skinematics.sensors.xsens import XSens

        initialPosition = array([0, 0, 0])
        R_initialOrientation = rotmat.R(0, 90)

        sensor = XSens(
            in_file=inFile,
            R_init=R_initialOrientation,
            pos_init=initialPosition,
            q_type="kalman",
        )
        self.assertEqual(sensor.quat.shape, (len(sensor.acc), 4))
        self.assertTrue(np.allclose(norm(sensor.quat, axis=1), 1))

    def test_madgwick(self):

        from skinematics.sensors.manual import MyOwnSensor

        ## Get data
        imu = self.imu_signals
        in_data = {
            "rate": imu["rate"],
            "acc": imu["gia"],
            "omega": imu["omega"],
            "mag": imu["magnetic"],
        }

        my_sensor = MyOwnSensor(
            in_file="Simulated sensor-data",
            in_data=in_data,
            R_init=quat.convert(self.q_init, to="rotmat"),
            pos_init=self.pos_init,
            q_type="madgwick",
        )

        # and then check, if the quat_vector = [0, sin(45), 0]
        q_madgwick = my_sensor.quat

        result = quat.q_vector(q_madgwick[-1])
        correct = array([0.0, np.sin(np.deg2rad(45)), 0.0])
        error = norm(result - correct)

        # self.assertAlmostEqual(error, 0)
        self.assertTrue(error < 1e-3)

        ##inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        ##from skinematics.sensors.xsens import XSens

        ##initialPosition = array([0,0,0])
        ##R_initialOrientation = rotmat.R(0,90)

        ##sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='madgwick')
        ##q = sensor.quat

    def test_mahony(self):

        from skinematics.sensors.manual import MyOwnSensor

        ## Get data
        imu = self.imu_signals
        in_data = {
            "rate": imu["rate"],
            "acc": imu["gia"],
            "omega": imu["omega"],
            "mag": imu["magnetic"],
        }

        my_sensor = MyOwnSensor(
            in_file="Simulated sensor-data",
            in_data=in_data,
            R_init=quat.convert(self.q_init, to="rotmat"),
            pos_init=self.pos_init,
            q_type="mahony",
        )

        # and then check, if the quat_vector = [0, sin(45), 0]
        q_mahony = my_sensor.quat

        result = quat.q_vector(q_mahony[-1])
        correct = array([0.0, np.sin(np.deg2rad(45)), 0.0])
        error = norm(result - correct)

        # self.assertAlmostEqual(error, 0)
        self.assertTrue(error < 1e-3)

        ### Get data
        ##inFile = os.path.join(myPath, 'data', 'data_xsens.txt')
        ##from skinematics.sensors.xsens import XSens

        ##initialPosition = array([0,0,0])
        ##R_initialOrientation = rotmat.R(0,90)

        ##sensor = XSens(in_file=inFile, R_init = R_initialOrientation, pos_init = initialPosition, q_type='mahony')
        ##q = sensor.quat

    @unittest.skipUnless(vqf_installed, 'requires "pip install scikit-kinematics[vqf]"')
    def test_vqf(self):

        from skinematics.sensors.manual import MyOwnSensor

        ## Get data
        imu = self.imu_signals
        in_data = {
            "rate": imu["rate"],
            "acc": imu["gia"],
            "omega": imu["omega"],
            "mag": imu["magnetic"],
        }
        correct = array([0.0, np.sin(np.deg2rad(45)), 0.0])

        # with magnetometer (9D), through the sensor-object
        my_sensor = MyOwnSensor(
            in_file="Simulated sensor-data",
            in_data=in_data,
            R_init=quat.convert(self.q_init, to="rotmat"),
            pos_init=self.pos_init,
            q_type="vqf",
        )

        # and then check, if the quat_vector = [0, sin(45), 0]
        q_vqf = my_sensor.quat
        error = norm(quat.q_vector(q_vqf[-1]) - correct)
        self.assertTrue(error < 1e-2)

        # without magnetometer (6D)
        del in_data["mag"]
        my_sensor = MyOwnSensor(
            in_file="Simulated sensor-data", in_data=in_data, q_type="vqf"
        )
        error = norm(quat.q_vector(my_sensor.quat[-1]) - correct)
        self.assertTrue(error < 1e-2)

        # offline variant
        q_offline = imus.vqf(
            imu["rate"], imu["gia"], imu["omega"], imu["magnetic"], offline=True
        )
        error = norm(quat.q_vector(q_offline[-1]) - correct)
        self.assertTrue(error < 1e-2)

    @unittest.skipUnless(vqf_installed, 'requires "pip install scikit-kinematics[vqf]"')
    def test_vqf_gyro_bias(self):
        """VQF estimates the gyroscope bias, so the orientation does not drift"""

        # 1 sec movement, followed by 9 sec rest
        imu_signals, body_pos_orient = simulate_imu(
            rate=100,
            t_move=1,
            t_total=10,
            q_init=self.q_init,
            rotation_axis=[0, 1, 0],
            deg=90,
            pos_init=self.pos_init,
            direction=[1, 0, 0],
            distance=0,
            B0=vector.normalize([1, 0, 1]),
        )

        # Add a constant gyroscope bias of 1 deg/s
        omega = imu_signals["omega"] + np.deg2rad([0, 0, 1])
        rate = imu_signals["rate"]

        def angle_error(q):
            """Rotation angle [deg] between the estimated and the true final orientation"""
            q_err = quat.q_mult(quat.q_inv(body_pos_orient["quat"][-1]), q[-1]).ravel()
            return np.rad2deg(2 * np.arccos(np.clip(np.abs(q_err[0]), 0, 1)))

        q_gyro = quat.calc_quat(omega, self.q_init, rate=rate, CStype="bf")
        q_vqf = imus.vqf(rate, imu_signals["gia"], omega, imu_signals["magnetic"])

        self.assertTrue(angle_error(q_gyro) > 5)  # pure integration drifts ...
        self.assertTrue(angle_error(q_vqf) < 2)  # ... VQF compensates the bias

    def test_IMU_calc_orientation_position(self):
        """Currently, this only tests if the two functions are running through"""

        # Get data, with a specified input from an XSens system
        # data_dir = resources.files('data')
        # inFile = data_dir/'data_xsens.txt'

        inFile = os.path.join(myPath, "data", "data_xsens.txt")
        initial_orientation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        initial_position = np.r_[0, 0, 0]

        from skinematics.sensors.xsens import XSens

        sensor = XSens(
            in_file=inFile, R_init=initial_orientation, pos_init=initial_position
        )
        sensor.calc_position()
        print("done")

    def test_set_qtype(self):
        """Tests if the test crashes on any of the existing qtype options"""

        # Get data
        # data_dir = resources.files('data')
        # inFile = data_dir/'data_xsens.txt'
        inFile = os.path.join(myPath, "data", "data_xsens.txt")
        from skinematics.sensors.xsens import XSens

        initialPosition = array([0, 0, 0])
        R_initialOrientation = rotmat.R(0, 90)

        sensor = XSens(
            in_file=inFile,
            R_init=R_initialOrientation,
            pos_init=initialPosition,
            q_type="kalman",
        )

        allowed_values = ["analytical", "kalman", "madgwick", "mahony", None]
        if vqf_installed:
            allowed_values.append("vqf")

        for sensor_type in allowed_values:
            print("{0} is running".format(sensor_type))
            sensor.set_qtype(sensor_type)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestSequenceFunctions(methodName="test_set_qtype"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # unittest.main()

    print("Thanks for using programs from Thomas!")
    sleep(0.2)
