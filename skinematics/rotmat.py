'''
Routines for working with rotation matrices
'''
 
'''
comment
author :  Thomas Haslwanter
date :    June-2017
'''
__version__ = '2.0'

import numpy as np
import sympy
from collections import namedtuple

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) ) 
    
from skinematics import quat

# For deprecation warnings
import deprecation
import warnings
#warnings.simplefilter('always', DeprecationWarning)

def R(axis=0, angle=90) :
    '''Rotation matrix for rotation about a cardinal axis.
    The argument is entered in degree.
    
    Parameters
    ----------
    axis : skalar
            Axis of rotation, has to be 0, 1, or 2
    alpha : float
            rotation angle [deg]

    Returns
    -------
    R : rotation matrix, for rotation about the specified axis

    Examples
    --------
    >>> rotmat.R(axis=0, alpha=45)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678],
           [ 0.        ,  0.70710678,  0.70710678]])
    
    >>> rotmat.R(axis=0)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        , -1.        ],
           [ 0.        ,  1.        ,  0.       ]])
    
    >>> rotmat.R(1, 45)
    array([[ 0.70710678,  0.        ,  0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710678,  0.        ,  0.70710678]])
    
    >>> rotmat.R(axis=2, alpha=45)
    array([[ 0.70710678, -0.70710678,  0.        ],
           [ 0.70710678,  0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])
    '''

    
    # convert from degrees into radian:
    a_rad = np.deg2rad(angle)
    
    if axis == 0:
        R = np.array([[1,             0,            0],
                      [0, np.cos(a_rad), -np.sin(a_rad)],
                      [0, np.sin(a_rad),  np.cos(a_rad)]])
        
    elif axis == 1:
        R = np.array([[ np.cos(a_rad), 0, np.sin(a_rad) ],
                      [            0,  1,             0 ],
                      [-np.sin(a_rad), 0, np.cos(a_rad) ]])
        
    elif axis == 2:
        R = np.array([[np.cos(a_rad), -np.sin(a_rad), 0],
                      [np.sin(a_rad),  np.cos(a_rad), 0],
                      [            0,             0,  1]])
    
    else:
        raise IOError('"axis" has to be 0, 1, or 2')
    return R

def R_s(axis=0, angle='alpha'):
    '''
    Symbolic rotation matrix about the given axis, by an angle with the given name 

    Returns
    -------
        R_symbolic : symbolic matrix for rotation about the given axis

    Examples
    --------

    >>> R_yaw = R_s(axis=2, angle='theta')
    
    >>> R_aero = R_s(2) * R_s(1) * R_s(0)

    '''

    alpha = sympy.Symbol(angle)

    if axis == 0:
        R_s =  sympy.Matrix([[1,                0,                 0],
                         [0, sympy.cos(alpha), -sympy.sin(alpha)],
                         [0, sympy.sin(alpha),  sympy.cos(alpha)]])
        
    elif axis == 1:
        R_s = sympy.Matrix([[sympy.cos(alpha),0, sympy.sin(alpha)],
                         [0,1,0],
                         [-sympy.sin(alpha), 0, sympy.cos(alpha)]])
        
    elif axis == 2:
        R_s = sympy.Matrix([[sympy.cos(alpha), -sympy.sin(alpha), 0],
                         [sympy.sin(alpha), sympy.cos(alpha), 0],
                         [0, 0, 1]])
    
    else:
        raise IOError('"axis" has to be 0, 1, or 2')
    return R_s

def sequence(R, to ='Euler'):
    '''
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
        - aero ... same as "Fick"
        - Helmholtz ... Ry * Rz * Rx

    Returns
    -------
    alpha : first rotation(right-most matrix)
    beta : second rotation 
    gamma : third rotation (left-most matrix)

    Notes
    -----
    The following formulas are used:

    Euler:
    
    .. math::
        \\beta   = -arcsin(\\sqrt{ R_{xz}^2 + R_{yz}^2 }) * sign(R_{yz})

        \\gamma = arcsin(\\frac{R_{xz}}{sin\\beta})

        \\alpha   = arcsin(\\frac{R_{zx}}{sin\\beta})
    
    aero / Fick:
    
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
    
    '''

    # Reshape the input such that I can use the standard matrix indices
    # For a simple (3,3) matrix, a superfluous first index is added.
    Rs = R.reshape((-1,3,3), order='C')
    
    if to=='Fick' or to=='aero':
        gamma =  np.arctan2(Rs[:,1,0], Rs[:,0,0])
        alpha =  np.arctan2(Rs[:,2,1], Rs[:,2,2])
        beta  = -np.arcsin(Rs[:,2,0])
    
    elif to == 'Helmholtz':
        gamma =  np.arcsin( Rs[:,1,0] )
        beta  = -np.arcsin( Rs[:,2,0]/np.cos(gamma) )
        alpha = -np.arcsin( Rs[:1,2]/np.cos(gamma) )
        
    elif to == 'Euler':
        epsilon = 1e-6
        beta = - np.arcsin(np.sqrt(Rs[:,0,2]**2 + Rs[:,1,2]**2)) * np.sign(Rs[:,1,2])
        small_indices =  beta < epsilon
        
        # For small beta
        beta[small_indices] = 0
        gamma[small_indices] = 0
        alpha[small_indices] = np.arcsin(Rs[small_indices,1,0])
        
        # for the rest
        gamma[~small_indices] = np.arcsin( Rs[~small_indices,0,2]/np.sin(beta) )
        alpha[~small_indices] = np.arcsin( Rs[~small_indices,2,0]/np.sin(beta) )
            
    else:
        print('\nSorry, only know: \naero, \nFick, \nHelmholtz, \nEuler.\n')
        raise IOError
        
    # Return the parameter-angles
    if R.size == 9:
        return np.r_[alpha, beta, gamma]
    else:
        return np.column_stack( (alpha, beta, gamma) )

# ------------ Everything below here is deprecated -------------------------------
@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R(0, angle)`` function instead")
def R1(psi):
    '''Rotation about the 1-axis.
    Deprecated, use R(0, psi)'''
    return R(axis=0, angle=psi)

@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R(1, angle)`` function instead")
def R2(phi):
    '''Rotation about the 2-axis.
    Deprecated, use R(1, phi)'''
    return R(axis=1, angle=phi)

@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R(2, angle)`` function instead")
def R3(theta):
    '''Rotation about the 3-axis.
    Deprecated, use R(2, theta)'''

@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``sequence(R, 'Euler')`` function instead")
def rotmat2Euler(R):
    '''
    This function takes a rotation matrix, and calculates the corresponding Euler-angles.
    Deprecated. Use "sequence(R, 'Euler')" '''
    
    return sequence(R, 'Euler')


@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``sequence(R, 'Fick')`` function instead")
def rotmat2Fick(R):
    '''
    This function takes a rotation matrix, and calculates the corresponding Fick-angles.
    Deprecated. Use "sequence(R, 'Fick')" '''

    return sequence(R, 'Fick')


@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``sequence(R, 'Helmholtz')`` function instead")
def rotmat2Helmholtz(R):
    '''
    This function takes a rotation matrix, and calculates the corresponding Helmholtz-angles.
    Deprecated. Use "sequence(R, 'Helmholtz')" '''
    return sequence(R, 'Helmholtz')


@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R_s(0, 'psi')`` function instead")
def R1_s():
    '''
    Symbolic rotation matrix about the 1-axis, by an angle psi 
    Deprecated. Use R_s(0, 'psi')'''
    return R_s(0, 'psi')

@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R_s(1, 'phi')`` function instead")
def R2_s():
    '''
    Symbolic rotation matrix about the 2-axis, by an angle phi 
    Deprecated. Use R_s(1, 'phi')'''
    return R_s(1, 'phi')
    
@deprecation.deprecated(deprecated_in="2.0", removed_in="2.2",
                        current_version=__version__,
                        details="Use the ``R_s(2, 'alpha')`` function instead")
def R3_s():
    '''
    Symbolic rotation matrix about the 3-axis, by an angle theta 
    Deprecated. Use R_s(2, 'theta')'''
    return R_s(2, 'theta')


def convert(rMat, to ='quat'):
    '''
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
    
    '''    
    
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
    

if __name__ == '__main__':
    testmat = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                   [np.sqrt(2)/2,  np.sqrt(2)/2, 0],
                   [0, 0, 1]])
    Fick = rotmat2Fick(testmat)
    print(Fick)
    
    angles = sequence(testmat, to='aero')
    print(angles)
    
    testmat2 = np.tile(np.reshape(testmat, (1,-1)), (2,1))
    angles = sequence(testmat2, to='aero')
    print(angles)
    
    print('Done testing')
    '''
    correct = np.r_[[0,0,np.pi/4]]
    print(R())
    print(R1(45))
    from pprint import pprint
    pprint(R_s(axis=0))
    pprint(R_s(axis=1, angle='phi'))
    
    print(R(1,45))
    print(R_s(1, 'gamma'))
    
    R2(30)
    print(R1(40))
    a = np.r_[np.cos(0.1), 0,0,np.sin(0.1)]
    print('The inverse of {0} is {1}'.format(a, quat.q_inv(a)))
    
    '''
