from skinematics import rotmat, quat
import unittest
import numpy as np

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.delta =  1e-5

    def test_seq2quat(self):
        angle = 10
        a = np.r_[angle, 0, 0]
        quats = rotmat.seq2quat([a])
        self.assertAlmostEqual(quats[0,3], np.sin(np.deg2rad(angle/2)))

        angles = np.array([[0,0,0],
                           [10,0,0],
                           [20,10,5]])
        angles2 = quat.quat2seq(rotmat.seq2quat(angles))
        error = np.sum((angles-angles2)**2)
        self.assertAlmostEqual(error,0)

    def test_R_axis0(self):
        R0 = np.array([[1,0,0],
                    [0, np.sqrt(2)/2, -np.sqrt(2)/2],
                    [0, np.sqrt(2)/2,  np.sqrt(2)/2]])

        self.assertTrue(np.all(np.abs(R0 - rotmat.R(0, 45))< self.delta))

    def test_R_axis1(self):
        R1 = np.array([[ np.sqrt(2)/2, 0, np.sqrt(2)/2],
                       [0, 1, 0],
                       [-np.sqrt(2)/2, 0, np.sqrt(2)/2]])

        self.assertTrue(np.all(np.abs(R1 - rotmat.R(1, 45))< self.delta))

    def test_R_axis2(self):
        R2 = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])

        self.assertTrue(np.all(np.abs(R2 - rotmat.R(2, 45))< self.delta))

    def test_symbolic(self):
        R_nautical = rotmat.R_s(2, 'theta')*rotmat.R_s(1, 'phi')*rotmat.R_s(0, 'psi')
        print(R_nautical)

    def test_Fick(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Fick = rotmat.sequence(testmat, to ='nautical')
        correct = np.r_[[np.pi/4, 0, 0]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Fick)), 0)

    def test_Helmholtz(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Helm = rotmat.sequence(testmat, to ='Helmholtz')
        correct = np.r_[[np.pi/4, 0, 0]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Helm)), 0)

    def test_Euler(self):
        testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                       [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                       [0, 0, 1]])
        Euler = rotmat.sequence(testmat, to ='Euler')
        correct = np.r_[[0, 0, np.pi/4]]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(Euler)), 0)

        alpha, beta, gamma = 10, 20, 30
        mat_euler = rotmat.R(2, gamma) @ rotmat.R(0, beta) @ rotmat.R(2, alpha)
        angles_euler = np.rad2deg(rotmat.sequence(mat_euler, to='Euler'))
        correct = np.r_[gamma, beta, alpha]
        self.assertAlmostEqual(np.linalg.norm(correct - np.array(angles_euler)), 0)

    def test_dh(self):
        theta = 45
        alpha = 45
        d=5
        r=10
        dh=  np.array([[np.sqrt(2) / 2,  -np.sqrt(2) / 2*np.sqrt(2) / 2,  np.sqrt(2) / 2*np.sqrt(2) / 2,  10*np.sqrt(2) / 2],
        [np.sqrt(2) / 2,  np.sqrt(2) / 2*np.sqrt(2) / 2 , -np.sqrt(2) / 2*np.sqrt(2) / 2,  10*np.sqrt(2) / 2],
        [0,           np.sqrt(2) / 2,             np.sqrt(2) / 2,             d],
        [0,               0,                        0,                        1]])
        print('dh',dh)
        print('rotmat.dh(theta,d,r,alpha)', rotmat.dh(theta,d,r,alpha))
        self.assertTrue(np.all(np.abs(dh - rotmat.dh(theta,d,r,alpha)) < self.delta))

    def test_dh_s(self):
        alpha = 40
        r=10
        print(rotmat.dh_s('theta','d',r,alpha))

    def test_convert(self):
        result = rotmat.convert(quat.convert([0, 0, 0.1], to ='rotmat'), to ='quat')
        correct = np.array([[ 0.99498744,  0.        ,  0.        ,  0.1       ]])
        error = np.linalg.norm(result - correct)
        self.assertTrue(error < self.delta)

if __name__ == '__main__':
    unittest.main()
