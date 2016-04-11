'''
Routines for working with rotation matrices
'''
 
'''
comment
author :  Thomas Haslwanter
date :    Dec-2014
version : 1.7
'''

import numpy as np
import sympy
from collections import namedtuple

def R1(psi):
    '''Rotation about the 1-axis.
    The argument is entered in degree.
    
    Parameters
    ----------
    psi : rotation angle about the 1-axis [deg]

    Returns
    -------
    R1 : rotation matrix, for rotation about the 1-axis

    Examples
    --------
    >>> rotmat.R1(45)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678],
           [ 0.        ,  0.70710678,  0.70710678]])
    
    '''

    
    # convert from degrees into radian:
    psi = psi * np.pi/180;
    
    R = np.array([[1, 0, 0],
            [0, np.cos(psi), -np.sin(psi)],
            [0, np.sin(psi),  np.cos(psi)]])
    
    return R

def R2(phi):
    '''Rotation about the 2-axis.
    The argument is entered in degree.
    
    Parameters
    ----------
    phi : rotation angle about the 2-axis [deg]

    Returns
    -------
    R2 : rotation matrix, for rotation about the 2-axis

    Examples
    --------
    >>> thLib.rotmat.R2(45)
    array([[ 0.70710678,  0.        ,  0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710678,  0.        ,  0.70710678]])
    
    '''

    
    # convert from degrees into radian:
    phi = phi * np.pi/180;
    
    R = np.array([[np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [  -np.sin(phi), 0, np.cos(phi)]])
    
    return R

def R3(theta):
    '''Rotation about the 3-axis.
    The argument is entered in degree.
    
    Parameters
    ----------
    theta : rotation angle about the 3-axis [deg]

    Returns
    -------
    R3 : rotation matrix, for rotation about the 3-axis

    Examples
    --------

    >>> thLib.rotmat.R3(45)
    array([[ 0.70710678, -0.70710678,  0.        ],
           [ 0.70710678,  0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    '''


    # convert from degrees into radian:
    theta = theta * np.pi/180;
    
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta),  np.cos(theta), 0],
               [0, 0, 1]])
    
    return R

def rotmat2Euler(R):
    '''
    This function takes a rotation matrix, and calculates
    the corresponding Euler-angles. 
    R_Euler = R3(gamma) * R1(beta) * R3(alpha)

    Parameters
    ----------
    R : rotation matrix

    Returns
    -------
    alpha : first rotation(about 3-axis)
    beta : second rotation (about 1-axis)
    gamma : third rotation (about 3-axis)

    Notes
    -----
    The following formulas are used:

    .. math::
        \\beta   = -arcsin(\\sqrt{ R_{13}^2 + R_{23}^2 }) * sign(R_{23})

        \\gamma = arcsin(\\frac{R_{13}}{sin\\beta})

        \\alpha   = arcsin(\\frac{R_{31}}{sin\\beta})
    
    '''

    epsilon = 1e-6
    beta = - np.arcsin(np.sqrt(R[0,2]**2 + R[1,2]**2)) * np.sign(R[1,2])
    if beta < 1e-6:
        beta = 0
        gamma = 0
        alpha = np.arcsin(R[1,0])
    else:
        gamma = np.arcsin(R[0,2]/np.sin(beta))
        alpha = np.arcsin(R[2,0]/np.cos(beta))

    Euler = namedtuple('Euler', ['alpha', 'beta', 'gamma'])
    return Euler(alpha, beta, gamma)

def rotmat2Fick(R):
    '''
    This function takes a rotation matrix, and calculates
    the corresponding Fick-angles. 

    Parameters
    ----------
    R : rotation matrix

    Returns
    -------
    psi : torsional  position (rotation about 1-axis)
    phi : vertical   position (rotation about 2-axis)
    theta : horizontal position (rotation about 3-axis)

    Notes
    -----
    The following formulas are used:

    .. math::

        \\theta   = arctan(\\frac{R_{21}}{R_{11}})

        \\phi = arcsin(R_{31})

        \\psi   = arctan(\\frac{R_{32}}{R_{33}})

    Note that it is assumed that psi < pi !
    '''

    theta = np.arctan2(R[1,0], R[0,0])
    psi = np.arctan2(R[2,1], R[2,2])
    phi = -np.arcsin(R[2,0])
    #phi = -np.arcsin(R[2,0])
    #theta = np.arcsin(R[1,0]/np.cos(phi))
    #psi = np.arcsin(R[2,1]/np.cos(phi))


    Fick = namedtuple('Fick', ['psi', 'phi', 'theta'])
    return Fick(psi, phi, theta)

def rotmat2Helmholtz(R):
    '''
    This function takes a rotation matrix, and calculates
    the corresponding Helmholtz-angles. 

    Parameters
    ----------
    R : rotation matrix

    Returns
    -------
    psi : torsional  position (rotation about 1-axis)
    phi : vertical   position (rotation about 2-axis)
    theta : horizontal position (rotation about 3-axis)

    Notes
    -----
    The following formulas are used:

    .. math::
        \\theta = arcsin(R_{21})

        \\phi = -arcsin(\\frac{R_{31}}{cos\\theta})

        \\psi = -arcsin(\\frac{R_{23}}{cos\\theta})


    Note that it is assumed that psi < pi
    
    '''

    theta = np.arcsin(R[1,0])
    phi = -np.arcsin(R[2,0]/np.cos(theta))
    psi = -np.arcsin(R[1,2]/np.cos(theta))

    Helm = namedtuple('Helm', ['psi', 'phi', 'theta'])
    return Helm(psi, phi, theta)


def R1_s():
    '''
    Symbolic rotation matrix about the 1-axis, by an angle psi 

    Returns
    -------
        R1_s : symbolic matrix for rotation about 1-axis

    Examples
    --------

    >>> R_Fick = R3_s() * R2_s() * R1_s()

    '''

    psi = sympy.Symbol('psi')

    return sympy.Matrix([[1,0,0],
                         [0, sympy.cos(psi), -sympy.sin(psi)],
                         [0, sympy.sin(psi), sympy.cos(psi)]])

def R2_s():
    '''
    Symbolic rotation matrix about the 2-axis, by an angle phi 

    Returns
    -------
        R2_s : symbolic matrix for rotation about 2-axis

    Examples
    --------

    >>> R_Fick = R3_s() * R2_s() * R1_s()

    '''

    phi = sympy.Symbol('phi')

    return sympy.Matrix([[sympy.cos(phi),0, sympy.sin(phi)],
                         [0,1,0],
                         [-sympy.sin(phi), 0, sympy.cos(phi)]])
    
def R3_s():
    '''
    Symbolic rotation matrix about the 3-axis, by an angle theta 

    Returns
    -------
        R3_s : symbolic matrix for rotation about 3-axis

    Examples
    --------

    >>> R_Fick = R3_s() * R2_s() * R1_s()

    '''

    theta = sympy.Symbol('theta')

    return sympy.Matrix([[sympy.cos(theta), -sympy.sin(theta), 0],
                         [sympy.sin(theta), sympy.cos(theta), 0],
                         [0, 0, 1]])


if __name__ == '__main__':
    testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                   [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                   [0, 0, 1]])
    Fick = rotmat2Fick(testmat)
    correct = np.r_[[0,0,np.pi/4]]
    print('Done testing')
    
