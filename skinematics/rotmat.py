"""
Routines for working with rotation matrices
"""

"""
comment
author :  Thomas Haslwanter
date :    April-2018
"""

import numpy as np
import sympy
from collections import namedtuple

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys

file_dir = os.path.dirname(__file__)
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    

# For deprecation warnings
#import deprecation
#import warnings
#warnings.simplefilter('always', DeprecationWarning)


def stm(axis='x', angle=0, translation=[0, 0, 0]):
    """Spatial Transformation Matrix
    
    Parameters
    ----------
    axis : int or str
            Axis of rotation, has to be 0, 1, or 2, or 'x', 'y', or 'z'
    angle : float
            rotation angle [deg]
    translation : 3x1 ndarray
            3D-translation vector

    Returns
    -------
    STM : 4x4 ndarray
        spatial transformation matrix, for rotation about the specified axis,
        and translation by the given vector

    Examples
    --------
    >>> rotmat.stm(axis='x', angle=45, translation=[1,2,3.3])
    array([[ 1.        ,  0.        ,  0.        ,  1.        ],
           [ 0.        ,  0.70710678, -0.70710678,  2.        ],
           [ 0.        ,  0.70710678,  0.70710678,  3.3       ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> R_z = rotmat.stm(axis='z', angle=30)
    >>> T_y = rotmat.stm(translation=[0, 10, 0])
    
    """
    axis = _check_axis(axis)

    stm = np.eye(4)
    stm[:-1,:-1] = R(axis, angle)
    stm[:3,-1] = translation
    return stm


def stm_s(axis='x', angle='0', transl='0,0,0'):
    """
    Symbolic spatial transformation matrix about the given axis, by an angle with
    the given name, and translation by the given distances.
    
    Parameters
    ----------
    axis : int or str
            Axis of rotation, has to be 0, 1, or 2, or 'x', 'y', or 'z'
    angle : string
            Name of rotation angle, or '0' for no rotation,
            'alpha', 'theta', etc. for a symbolic rotation.
    transl : string
            Has to contain three names, for the three translation distances.
            '0,0,0' would correspond to no translation, and
            'x,y,z' to an arbitrary translation.


    Returns
    -------
        STM_symbolic : corresponding symbolic spatial transformation matrix 

    Examples
    --------

    >>> Rz_s = STM_s(axis='z', angle='theta', transl='0,0,0')
    
    >>> Tz_s = STM_s(axis=0, angle='0', transl='0,0,z')

    """
    axis = _check_axis(axis)

    # Default is the unit matrix
    STM_s = sympy.eye(4)
    
    if angle != '0':
        STM_s[:3,:3] = R_s(axis, angle)
        
    transl = transl.replace(' ', '')
    if not transl==('0,0,0'):
        trans_dir = transl.split(',')
        assert(len(trans_dir)==3)
        for (ii, magnitude) in enumerate(trans_dir):
            if magnitude != '0':
                symbol = sympy.Symbol(magnitude)
                STM_s[ii,-1] = symbol
            
    return STM_s        
        
    
def R(axis='x', angle=90) :
    """Rotation matrix for rotation about a cardinal axis.
    The argument is entered in degree.
    
    Parameters
    ----------
    axis : int or str
            Axis of rotation, has to be 0, 1, or 2, or 'x', 'y', or 'z'
    angle : float
            rotation angle [deg]

    Returns
    -------
    R : rotation matrix, for rotation about the specified axis

    Examples
    --------
    >>> rotmat.R(axis='x', angle=45)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678],
           [ 0.        ,  0.70710678,  0.70710678]])
    
    >>> rotmat.R(axis='x')
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        , -1.        ],
           [ 0.        ,  1.        ,  0.       ]])
    
    >>> rotmat.R('y', 45)
    array([[ 0.70710678,  0.        ,  0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710678,  0.        ,  0.70710678]])
    
    >>> rotmat.R(axis=2, angle=45)
    array([[ 0.70710678, -0.70710678,  0.        ],
           [ 0.70710678,  0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])
    """

    axis = _check_axis(axis)

    # convert from degrees into radian:
    a_rad = np.deg2rad(angle)
    
    if axis == 'x':
        R = np.array([[1,             0,            0],
                      [0, np.cos(a_rad), -np.sin(a_rad)],
                      [0, np.sin(a_rad),  np.cos(a_rad)]])
        
    elif axis == 'y':
        R = np.array([[ np.cos(a_rad), 0, np.sin(a_rad) ],
                      [            0,  1,             0 ],
                      [-np.sin(a_rad), 0, np.cos(a_rad) ]])
        
    elif axis == 'z':
        R = np.array([[np.cos(a_rad), -np.sin(a_rad), 0],
                      [np.sin(a_rad),  np.cos(a_rad), 0],
                      [            0,             0,  1]])
    
    else:
        raise IOError('"axis" has to be "x", "y", or "z"')
    return R


def _check_axis(sel_axis):
    """Leaves u[x/y/z] nchanged, but converts [0/1/2] to [x/y/z]

    Parameters
    ----------
        sel_axis : str or int
            If "str", the value has to be 'x', 'y', or 'z'
            If "int", the value has to be 0, 1, or 2

    Returns
    -------
        axis : str
        Selected axis, as string
    """

    seq = 'xyz'
    if type(sel_axis) is str:
        if sel_axis not in seq:
            raise IOError
        axis = sel_axis
    elif type(sel_axis) is int:
        if sel_axis in range(3):
            axis = seq[sel_axis]
        else:
            raise IOError

    return axis


def R_s(axis='x', angle='alpha'):
    """
    Symbolic rotation matrix about the given axis, by an angle with the given name 
    
    Parameters
    ----------
    axis : int or str
            Axis of rotation, has to be 0, 1, or 2, or 'x', 'y', or 'z'
    alpha : string
            name of rotation angle

    Returns
    -------
        R_symbolic : symbolic matrix for rotation about the given axis

    Examples
    --------

    >>> R_yaw = R_s(axis=2, angle='theta')
    
    >>> R_nautical = R_s(2) * R_s(1) * R_s(0)

    """

    axis = _check_axis(axis)

    alpha = sympy.Symbol(angle)

    if axis == 'x':
        R_s =  sympy.Matrix([[1,                0,                 0],
                         [0, sympy.cos(alpha), -sympy.sin(alpha)],
                         [0, sympy.sin(alpha),  sympy.cos(alpha)]])
        
    elif axis == 'y':
        R_s = sympy.Matrix([[sympy.cos(alpha),0, sympy.sin(alpha)],
                         [0,1,0],
                         [-sympy.sin(alpha), 0, sympy.cos(alpha)]])
        
    elif axis == 'z':
        R_s = sympy.Matrix([[sympy.cos(alpha), -sympy.sin(alpha), 0],
                         [sympy.sin(alpha), sympy.cos(alpha), 0],
                         [0, 0, 1]])
    
    else:
        raise IOError('"axis" has to be "x", "y", or "z", not {0}'.format(axis))
    return R_s


def sequence(R, to ='Euler'):
    """
    This function takes a rotation matrix, and calculates
    the corresponding angles for sequential rotations. 
    
    R_Euler = R3(gamma) * R1(beta) * R3(alpha)

    Parameters
    ----------
    R : ndarray, 3x3
        rotation matrix
    to : string
        Has to be one the following:
        
        - Euler ... Rz * Rx * Rz
        - Fick ... Rz * Ry * Rx
        - nautical ... same as "Fick"
        - Helmholtz ... Ry * Rz * Rx

    Returns
    -------
    gamma : third rotation (left-most matrix)
    beta : second rotation 
    alpha : first rotation(right-most matrix)

    Notes
    -----
    The following formulas are used:

    Euler:
    
    .. math::
        \\beta   = -arcsin(\\sqrt{ R_{xz}^2 + R_{yz}^2 }) * sign(R_{yz})

        \\gamma = arcsin(\\frac{R_{xz}}{sin\\beta})

        \\alpha   = arcsin(\\frac{R_{zx}}{sin\\beta})
    
    nautical / Fick:
    
    .. math::

        \\theta   = arctan(\\frac{R_{yx}}{R_{xx}})

       \\phi = arcsin(R_{zx})

        \\psi   = arctan(\\frac{R_{zy}}{R_{zz}})

    Note that it is assumed that psi < pi !
    
    Helmholtz: 
    
    .. math::
        \\theta = arcsin(R_{yx})

        \\phi = -arcsin(\\frac{R_{zx}}{cos\\theta})

        \\psi = -arcsin(\\frac{R_{yz}}{cos\\theta})


    Note that it is assumed that psi < pi
    
    """

    # Reshape the input such that I can use the standard matrix indices
    # For a simple (3,3) matrix, a superfluous first index is added.
    Rs = R.reshape((-1,3,3), order='C')
    
    if to=='Fick' or to=='nautical':
        gamma =  np.arctan2(Rs[:,1,0], Rs[:,0,0])
        alpha =  np.arctan2(Rs[:,2,1], Rs[:,2,2])
        beta  = -np.arcsin(Rs[:,2,0])
    
    elif to == 'Helmholtz':
        gamma =  np.arcsin( Rs[:,1,0] )
        beta  = -np.arcsin( Rs[:,2,0]/np.cos(gamma) )
        alpha = -np.arcsin( Rs[:,1,2]/np.cos(gamma) )
        
    elif to == 'Euler':
        epsilon = 1e-6
        beta = - np.arcsin(np.sqrt(Rs[:,0,2]**2 + Rs[:,1,2]**2)) * np.sign(Rs[:,1,2])
        small_indices =  beta < epsilon
        
        # Assign memory for alpha and gamma
        alpha = np.nan * np.ones_like(beta)
        gamma = np.nan * np.ones_like(beta)
        
        # For small beta
        beta[small_indices] = 0
        gamma[small_indices] = 0
        alpha[small_indices] = np.arcsin(Rs[small_indices,1,0])
        
        # for the rest
        gamma[~small_indices] = np.arcsin( Rs[~small_indices,0,2]/np.sin(beta) )
        alpha[~small_indices] = np.arcsin( Rs[~small_indices,2,0]/np.sin(beta) )
            
    else:
        print('\nSorry, only know: \nnautical, \nFick, \nHelmholtz, \nEuler.\n')
        raise IOError
        
    # Return the parameter-angles
    if R.size == 9:
        return np.r_[gamma, beta, alpha]
    else:
        return np.column_stack( (gamma, beta, alpha) )

    
def dh(theta=0, d=0, r=0, alpha=0):
    """
    Denavit Hartenberg transformation and rotation matrix.

    .. math::
        T_n^{n - 1}= {Trans}_{z_{n - 1}}(d_n)
        \\cdot {Rot}_{z_{n - 1}}(\\theta_n) \\cdot {Trans}_{x_n}(r_n) \\cdot {Rot}_{x_n}(\\alpha_n)


    .. math::
        T_n=\\left[\\begin{array}{ccc|c}
        \\cos\\theta_n & -\\sin\\theta_n \\cos\\alpha_n & \\sin\\theta_n \\sin\\alpha_n & r_n\\cos\\theta_n \\\\
        \\sin\\theta_n & \\cos\\theta_n \\cos\\alpha_n & -\\cos\\theta_n \\sin\\alpha_n & r_n \\sin\\theta_n \\\\
        0 & \\sin\\alpha_n & \\cos\\alpha_n & d_n \\\\
        \\hline
        0 & 0 & 0 & 1
        \\end{array}
        \\right] =\\left[\\begin{array}{ccc|c}
        & & & \\\\
        & R & & T \\\\
        & & & \\\\
        \\hline
        0 & 0 & 0 & 1
        \\end{array}\\right]


    Examples
    --------

    >>> theta_1=90.0
    >>> theta_2=90.0
    >>> theta_3=0.
    >>> dh(theta_1,60,0,0)*dh(0,88,71,90)*dh(theta_2,15,0,0)*dh(0,0,174,-180)*dh(theta_3,15,0,0)
    [[-6.12323400e-17 -6.12323400e-17 -1.00000000e+00 -7.10542736e-15],
    [ 6.12323400e-17  1.00000000e+00 -6.12323400e-17  7.10000000e+01],
    [  1.00000000e+00 -6.12323400e-17 -6.12323400e-17  3.22000000e+02],
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

    Parameters
    ----------
    theta : float
        rotation angle z axis [deg]

    d : float
        transformation along the z-axis

    alpha : float
        rotation angle x axis [deg]

    r : float
        transformation along the x-axis


    Returns
    -------
    dh : ndarray(4x4)
        Denavit Hartenberg transformation matrix.

    """
    
    # Calculate Denavit-Hartenberg transformation matrix
    Rx = stm(axis=0, angle=alpha)
    Tx = stm(translation=[r, 0, 0])
    Rz = stm(axis=2, angle=theta)
    Tz = stm(translation=[0, 0, d])
    
    t_dh = Tz @ Rz @ Tx @ Rx    
    
    return(t_dh)


def dh_s(theta=0, d=0, r=0, alpha=0):
    """
    Symbolic Denavit Hartenberg transformation and rotation matrix.


    >>> dh_s('theta_1',60,0,0)*dh_s(0,88,71,90)*dh_s('theta_2',15,0,0)*dh_s(0,0,174,-180)*dh_s('theta_3',15,0,0)

    Parameters
    ----------
    theta : float
        rotation angle z axis [deg]

    d : float
        transformation along the z-axis

    alpha : float
        rotation angle x axis [deg]

    r : float
        transformation along the x-axis



    Returns
    -------
    R : Symbolic rotation and transformation  matrix 4x4
    """
    
    # Force the correct input type
    theta_s = str(theta)
    d_s     = str(d)
    r_s     = str(r)
    alpha_s = str(alpha)
        
    # Calculate Denavit-Hartenberg transformation matrix
    Rx = stm_s(axis=0, angle = alpha_s)
    Tx = stm_s(transl = r_s + ',0,0')
    Rz = stm_s(axis=2, angle = theta_s)
    Tz = stm_s(transl='0,0,' + d_s)
    
    t_dh = Tz * Rz * Tx * Rx    
    
    return(t_dh)


def convert(rMat, to ='quat'):
    """
    Converts a rotation matrix to the corresponding quaternion.
    Assumes that R has the shape (3,3), or the matrix elements in columns

    Parameters
    ----------
    rMat : array, shape (3,3) or (N,9)
        single rotation matrix, or matrix with rotation-matrix elements.
    to : string
        Currently, only 'quat' is supported
    
    Returns
    -------
    outQuat : array, shape (4,) or (N,4)
        corresponding quaternion vector(s)
    
    Notes
    -----

    .. math::
         \\vec q = 0.5*copysign\\left( {\\begin{array}{*{20}{c}}
        {\\sqrt {1 + {R_{xx}} - {R_{yy}} - {R_{zz}}} ,}\\\\
        {\\sqrt {1 - {R_{xx}} + {R_{yy}} - {R_{zz}}} ,}\\\\
        {\\sqrt {1 - {R_{xx}} - {R_{yy}} + {R_{zz}}} ,}
        \\end{array}\\begin{array}{*{20}{c}}
        {{R_{zy}} - {R_{yz}}}\\\\
        {{R_{xz}} - {R_{zx}}}\\\\
        {{R_{yx}} - {R_{xy}}}
        \\end{array}} \\right) 
    
    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    
    >>> rotMat = array([[cos(alpha), -sin(alpha), 0],
    >>>    [sin(alpha), cos(alpha), 0],
    >>>    [0, 0, 1]])
    >>> rotmat.convert(rotMat, 'quat')
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342]])
    
    """
    
    if to != 'quat':
        raise IOError('Only know "quat"!')
    
    if rMat.shape == (3,3) or rMat.shape == (9,):
        rMat=np.atleast_2d(rMat.ravel()).T
    else:
        rMat = rMat.T
    q = np.zeros((4, rMat.shape[1]))
    
    R11 = rMat[0]
    R12 = rMat[1]
    R13 = rMat[2]
    R21 = rMat[3]
    R22 = rMat[4]
    R23 = rMat[5]
    R31 = rMat[6]
    R32 = rMat[7]
    R33 = rMat[8]
    
    q[1] = 0.5 * np.copysign(np.sqrt(1+R11-R22-R33), R32-R23)
    q[2] = 0.5 * np.copysign(np.sqrt(1-R11+R22-R33), R13-R31)
    q[3] = 0.5 * np.copysign(np.sqrt(1-R11-R22+R33), R21-R12)
    q[0] = np.sqrt(1-(q[1]**2+q[2]**2+q[3]**2))
    
    return q.T
    

def seq2quat(rot_angles, seq='nautical'):
    """
    This function takes a angles from sequenctial rotations  and calculates
    the corresponding quaternions.
    
    Parameters
    ----------
    rot_angles : ndarray, nx3
        sequential rotation angles [deg]
    seq : string
        Has to be one the following:
        
        - Euler ... Rz * Rx * Rz
        - Fick ... Rz * Ry * Rx
        - nautical ... same as "Fick"
        - Helmholtz ... Ry * Rz * Rx

    Returns
    -------
    quats : ndarray, nx4
        corresponding quaternions

    Examples
    --------
    >>> skin.rotmat.seq2quat([90, 23.074, -90], seq='Euler')
    array([[ 0.97979575,  0.        ,  0.2000007 ,  0.        ]])
    
    Notes
    -----
    The equations are longish, and can be found in 3D-Kinematics, 4.1.5 "Relation to Sequential Rotations"

    """
    
    rot_angles = np.atleast_2d(np.deg2rad(rot_angles))
    
    quats = np.nan * np.ones( [rot_angles.shape[0], 4] )
    
    if seq =='Fick' or seq =='nautical':
        theta = rot_angles[:,0]
        phi = rot_angles[:,1]
        psi = rot_angles[:,2]
        
        c_th, s_th = np.cos(theta/2), np.sin(theta/2)
        c_ph, s_ph = np.cos(phi/2),   np.sin(phi/2)
        c_ps, s_ps = np.cos(psi/2),   np.sin(psi/2)
        
        quats[:,0] = c_th*c_ph*c_ps + s_th*s_ph*s_ps
        quats[:,1] = c_th*c_ph*s_ps - s_th*s_ph*c_ps
        quats[:,2] = c_th*s_ph*c_ps + s_th*c_ph*s_ps
        quats[:,3] = s_th*c_ph*c_ps - c_th*s_ph*s_ps
        
    elif seq == 'Helmholtz':
        phi = rot_angles[:,0]
        theta = rot_angles[:,1]
        psi = rot_angles[:,2]
        
        c_th, s_th = np.cos(theta/2), np.sin(theta/2)
        c_ph, s_ph = np.cos(phi/2),   np.sin(phi/2)
        c_ps, s_ps = np.cos(psi/2),   np.sin(psi/2)
        
        quats[:,0] = c_th*c_ph*c_ps - s_th*s_ph*s_ps
        quats[:,1] = c_th*c_ph*s_ps + s_th*s_ph*c_ps
        quats[:,2] = c_th*s_ph*c_ps + s_th*c_ph*s_ps
        quats[:,3] = s_th*c_ph*c_ps - c_th*s_ph*s_ps
        
    elif seq == 'Euler':
        alpha = rot_angles[:,0]
        beta = rot_angles[:,1]
        gamma = rot_angles[:,2]
        
        c_al, s_al = np.cos(alpha/2), np.sin(alpha/2)
        c_be, s_be = np.cos(beta/2),  np.sin(beta/2)
        c_ga, s_ga = np.cos(gamma/2), np.sin(gamma/2)
        
        quats[:,0] = c_al*c_be*c_ga - s_al*c_be*s_ga
        quats[:,1] = c_al*s_be*c_ga + s_al*s_be*s_ga
        quats[:,2] = s_al*s_be*c_ga - c_al*s_be*s_ga
        quats[:,3] = c_al*c_be*s_ga + s_al*c_be*c_ga
        
    else:
        raise ValueError('Input parameter {0} not known'.format(seq))
    
    return quats
        
if __name__ == '__main__':
    from pprint import pprint
    
    STM = stm(axis=0, angle=45, translation=[1, 2, 3.3])
    R_z = stm(axis=2, angle=30)
    T_y = stm(translation=[0, 10., 0])
    
    pprint(STM)
    pprint(R_z)
    pprint(T_y)
    
    out_s = stm_s(axis=0, angle='0', transl='x,0,z')
    pprint(out_s)
    Rx = stm_s(axis=0, angle='alpha')
    Tx = stm_s(transl='r,0,0')
    Rz = stm_s(axis=2, angle='theta')
    Tz = stm_s(transl='0,0,d')
    dh_mat = Tz * Rz * Tx * Rx
    pprint(Rx)
    pprint(Tx)
    pprint(Rz)
    pprint(Tz)
    pprint(dh_mat)
    
    
    Rx = stm_s(axis=0, angle='0')
    Tx = stm_s(transl='0,0,0')
    Rz = stm_s(axis=2, angle='theta')
    Tz = stm_s(transl='0,0,15')
    dh2 = Tz * Rz * Tx * Rx    
    pprint(dh2)
    
    pprint(dh_s('theta', 15, 0, 0))
    
    t_dh = dh(60,15,0,0)
    print(t_dh)
    
    
    angles = np.r_[20, 0, 0]
    quat = seq2quat(angles)
    print(quat)
    testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                   [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                   [0, 0, 1]])
    
    angles = sequence(testmat, to='nautical')
    print(angles)
    
    testmat2 = np.tile(np.reshape(testmat, (1,-1)), (2,1))
    angles = sequence(testmat2, to='nautical')
    print(angles)
    
    print('Done testing')
    correct = np.r_[[0,0,np.pi/4]]
    print(R_s(1, 'gamma'))

    import quat
    a = np.r_[np.cos(0.1), 0,0,np.sin(0.1)]
    print('The inverse of {0} is {1}'.format(a, quat.q_inv(a)))
    
    print(R(1, 45))
    print(R('y', 45))
