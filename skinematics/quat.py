'''
Functions for working with quaternions. Note that all the functions also
work on arrays, and can deal with full quaternions as well as with
quaternion vectors.

A "Quaternion" class is defined, with

- operator overloading for mult, div, and inv.
- indexing

'''

'''
author: Thomas Haslwanter
date:   March 2017
ver:    0.3
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# The following construct is required since I want to run the module as a script
# inside the thLib-directory
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from skinematics import rotmat

pi = np.pi

class Quaternion():
    '''Quaternion class, with multiplocation, division, and inverse.
    A Quaternion can be created from vectors, rotation matrices,
    or from Fick-angles, Helmholtz-angles, or Euler angles (in deg).
    It provides
    
    * operator overloading for mult, div, and inv.
    * indexing
    * access to the data, in the attribute *values*.

    Parameters
    ----------
    inData : ndarray
             Contains the data in one of the following formats:
             
             * vector : (3 x n) or (4 x n) array, containing the quaternion values
             * rotmat : array, shape (3,3) or (N,9)
                        single rotation matrix, or matrix with rotation-matrix elements.
             * Fick : (3 x n) array, containing (psi, phi, theta) rotations about
                                the (1,2,3) axes [deg] (Fick sequence)
             * Helmholtz : (3 x n) array, containing (psi, phi, theta) rotations about
                                the (1,2,3) axes [deg] (Helmholtz sequence)
             * Euler : (3 x n) array, containing (alpha, beta, gamma) rotations about
                                the (3,1,3) axes [deg] (Euler sequence)
    inType : string
           Specifies the type of the input and has to have one of the following values
           'vector'[Default], 'rotmat', 'Fick', 'Helmholtz', 'Euler'


    Attributes
    ----------
    
    values : (4 x n) array
        quaternion values
             

    Methods
    -------
    
    inv()
        Inverse of the quaterion
    export(to='rotmat')
        Export to one of the following formats: 'rotmat', 'Euler', 'Fick', 'Helmholtz'
    
    Notes
    -----
    .. math::
          \\vec {q}_{Euler}  = \\left[ {\\begin{array}{*{20}{c}}
          {\\cos \\frac{\\alpha }{2}*\\cos \\frac{\\beta }{2}*\\cos \\frac{\\gamma }{2} - \\sin \\frac{\\alpha }{2}\\cos \\frac{\\beta }{2}\\sin \\frac{\\gamma }{2}} \\\\ 
          {\\cos \\frac{\\alpha }{2}*\\sin \\frac{\\beta }{2}*\\cos \\frac{\\gamma }{2} + \\sin \\frac{\\alpha }{2}\\sin \\frac{\\beta }{2}\\sin \\frac{\\gamma }{2}} \\\\ 
          {\\cos \\frac{\\alpha }{2}*\\sin \\frac{\\beta }{2}*\\sin \\frac{\\gamma }{2} - \\sin \\frac{\\alpha }{2}\\sin \\frac{\\beta }{2}\\cos \\frac{\\gamma }{2}} \\\\ 
          {\\cos \\frac{\\alpha }{2}*\\cos \\frac{\\beta }{2}*\\sin \\frac{\\gamma }{2} + \\sin \\frac{\\alpha }{2}\\cos \\frac{\\beta }{2}\\cos \\frac{\\gamma }{2}} 
        \\end{array}} \\right]
        
    .. math::
          \\vec {q}_{Fick}  = \\left[ {\\begin{array}{*{20}{c}}
          {\\cos \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} + \\sin \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\sin \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} - \\cos \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\cos \\frac{\\psi }{2}*\\sin \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} + \\sin \\frac{\\psi }{2}\\cos \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\cos \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\sin \\frac{\\theta }{2} - \\sin \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\cos \\frac{\\theta }{2}} 
        \\end{array}} \\right]
        
    .. math::
          \\vec {q}_{Helmholtz}  = \\left[ {\\begin{array}{*{20}{c}}
          {\\cos \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} - \\sin \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\sin \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} + \\cos \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\cos \\frac{\\psi }{2}*\\sin \\frac{\\phi }{2}*\\cos \\frac{\\theta }{2} + \\sin \\frac{\\psi }{2}\\cos \\frac{\\phi }{2}\\sin \\frac{\\theta }{2}} \\\\ 
          {\\cos \\frac{\\psi }{2}*\\cos \\frac{\\phi }{2}*\\sin \\frac{\\theta }{2} - \\sin \\frac{\\psi }{2}\\sin \\frac{\\phi }{2}\\cos \\frac{\\theta }{2}} 
        \\end{array}} \\right]

    Examples
    --------

    >>> q = Quaternion(array([[0,0,0.1],
                              [0,0,0.2],
                              [0,0,0.5]]))
    >>> p = Quaternion(array([0,0,0.2]))
    >>> fick = Quaternion( array([[0,0,10],
                                  [0,10,10]]), 'Fick')
    >>> combined = p * q
    >>> divided = q / p
    >>> extracted = q[1:2]
    >>> len(q)
    >>> data = q.values
    >>> 2
    >>> inv(q)

    '''
    
    def __init__(self, inData, inType='vector'):    
        '''Initialization'''

        if inType.lower() == 'vector':
            if isinstance(inData, np.ndarray) or isinstance(inData, list):
                self.values = vect2quat(inData)
            elif isinstance(inData, Quaternion):
                self.values = inData.values
            else:
                raise TypeError('Quaternions can only be based on ndarray or Quaternions!')
        
        elif inType.lower() == 'rotmat':
            '''Conversion from rotation matrices to quaternions.'''
            self.values = rotmat2quat(inData)
            
        elif inType.lower() == 'euler':
            ''' Conversion from Euler angles to quaternions.
            (a,b,g) stands for (alpha, beta, gamma) '''
            
            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)
            
            (ca, cb, cg) = np.cos(inData.T)
            (sa, sb, sg) = np.sin(inData.T)
            
            self.values = np.vstack( (ca*cb*cg - sa*cb*sg,
                                      ca*sb*cg + sa*sb*sg,
                                      ca*sb*sg - sa*sb*cg,
                                      ca*cb*sg + sa*cb*cg) ).T
        elif inType.lower() == 'fick':
            ''' Conversion from Fick angles to quaternions.
            (p,f,t) stands for (psi, phi, theta) '''
            
            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)
            
            (cp, cf, ct) = np.cos(inData.T)
            (sp, sf, st) = np.sin(inData.T)
            
            self.values = np.vstack( (cp*cf*ct + sp*sf*st,
                                      sp*cf*ct - cp*sf*st,
                                      cp*sf*ct + sp*cf*st,
                                      cp*cf*st - sp*sf*ct) ).T
            
        elif inType.lower() == 'helmholtz':
            ''' Conversion from Helmholtz angles to quaternions.
            (p,f,t) stands for (psi, phi, theta) '''
            
            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)
            
            (cp, cf, ct) = np.cos(inData.T)
            (sp, sf, st) = np.sin(inData.T)
            
            self.values = np.vstack( (cp*cf*ct - sp*sf*st,
                                      sp*cf*ct + cp*sf*st,
                                      cp*sf*ct + sp*cf*st,
                                      cp*cf*st - sp*sf*ct ) ).T
        
    def __len__(self):
        '''The "length" is given by the number of quaternions.'''
        return len(self.values)
    
    def __mul__(self, other):
        '''Operator overloading for multiplication.'''
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values * other)
        else:
            return Quaternion(quatmult(self.values, other.values))
    
    def __div__(self, other):
        '''Operator overloading for division.'''
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values / other)
        else:
            return Quaternion(quatmult(self.values, quatinv(other.values)))
    
    def __truediv__(self, other):
        '''Operator overloading for division.'''
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values / other)
        else:
            return Quaternion(quatmult(self.values, quatinv(other.values)))
    
    def __getitem__(self, select):
        return Quaternion(self.values[select])
        
    def __setitem__(self, select, item):
        self.values[select] = vect2quat(item)
        
    #def __delitem__(self, select):
        #np.delete(self.values, select, axis=0)
        
    def inv(self):
        '''Inverse of a quaternion.'''
        return Quaternion(quatinv(self.values))
    
    def __repr__(self):
        return 'Quaternion ' + str(self.values)
    
    def export(self, to='rotmat'):
        '''
        Conversion to other formats. May be slow for "Fick", "Helmholtz", and "Euler".
        
        Parameters
        ----------
        to : string
            content of returned values
            
            * 'rotmat' : rotation matrices (default), each flattened to a 9-dim vector
            * 'Euler' : Euler angles
            * 'Fick' : Fick angles
            * 'Helmholtz' : Helmholtz angles
            * 'vector' : vector part of the quaternion
        
        Returns
        -------
        ndarray, with the specified content


        Examples
        --------
        
        >>> q = Quaternion([0,0.2,0.1])
        >>> rm = q.export()
        >>> fick = q.export('Fick')
        
        '''
        if to.lower() == 'rotmat' :
           return quat2rotmat(self.values)
       
        if to.lower() == 'vector' :
            return self.values[:,1:]

        if to.lower() == 'euler':
            Euler = np.zeros((len(self),3))
            rm = self.export()
            if rm.shape == (3,3):
                rm = rm.reshape((1,9))
            for ii in range(len(self)):
               Euler[ii,:] = rotmat.rotmat2Euler(rm[ii].reshape((3,3)))
            return Euler
        
        if to.lower() == 'fick':
            Fick = np.zeros((len(self),3))
            rm = self.export()
            if rm.shape == (3,3):
                rm = rm.reshape((1,9))
            for ii in range(len(self)):
               Fick[ii,:] = rotmat.rotmat2Fick(rm[ii].reshape((3,3)))
            return Fick
        
        if to.lower() == 'helmholtz':
            Helmholtz = np.zeros((len(self),3))
            rm = self.export()
            if rm.shape == (3,3):
                rm = rm.reshape((1,9))
            for ii in range(len(self)):
               Helmholtz[ii,:] = rotmat.rotmat2Helmholtz(rm[ii].reshape((3,3)))
            return Helmholtz
        
def deg2quat(inDeg):
    '''
    Convert axis-angles or plain degree into the corresponding quaternion values.
    Can be used with a plain number or with an axis angle.

    Parameters
    ----------
    inDeg : float or (N,3)
        quaternion magnitude or quaternion vectors.
    
    Returns
    -------
    outQuat : float or array (N,3)
        number or quaternion vector.
    
    Notes
    -----
    
    .. math::
        | \\vec{q} | = sin(\\theta/2)

    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> quat.deg2quat(array([[10,20,30], [20,30,40]]))
    array([[ 0.08715574,  0.17364818,  0.25881905],
       [ 0.17364818,  0.25881905,  0.34202014]])

    >>> quat.deg2quat(10)
    0.087155742747658166

    '''
    deg = (inDeg+180)%360-180
    return np.sin(0.5 * deg * pi/180)
    
def quatconj(q):
    ''' Conjugate quaternion 
    
    Parameters
    ----------
    q: array_like, shape ([3,4],) or (N,[3/4])
        quaternion or quaternion vectors
    
    Returns
    -------
    qconj : conjugate quaternion(s)
    
    
    Examples
    --------
    >>>  quat.quatconj([0,0,0.1])
    array([ 0., -0., -0., -1.])
    
    >>> quat.quatconj([[cos(0.1),0,0,sin(0.1)],
    >>>    [cos(0.2), 0, sin(0.2), 0]])
    array([[ 0.99500417, -0.        , -0.        , -0.09983342],
           [ 0.98006658, -0.        , -0.19866933, -0.        ]])
    
    '''
    
    q = np.atleast_2d(q)
    if q.shape[1]==3:
        q = vect2quat(q)

    qConj = q * np.r_[1, -1,-1,-1]

    if q.shape[0]==1:
        qConj=qConj.ravel()

    return qConj

def quatinv(q):
    ''' Quaternion inversion 

    Parameters
    ----------
    q: array_like, shape ([3,4],) or (N,[3/4])
        quaternion or quaternion vectors
    
    Returns
    -------
    qinv : inverse quaternion(s)
    
    Notes
    -----

    .. math::
          q^{-1} = \\frac{q_0 - \\vec{q}}{|q|^2}

    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>>  quat.quatinv([0,0,0.1])
    array([[-0. , -0. , -0.1]])
    
    >>> quat.quatinv([[cos(0.1),0,0,sin(0.1)],
    >>> [cos(0.2),0,sin(0.2),0]])
    array([[ 0.99500417, -0.        , -0.        , -0.09983342],
           [ 0.98006658, -0.        , -0.19866933, -0.        ]])
    '''
    
    q = np.atleast_2d(q)
    if q.shape[1]==3:
        return -q
    else:
        qLength = np.sum(q**2, 1)
        qConj = q * np.r_[1, -1,-1,-1]
        return (qConj.T / qLength).T

def quatmult(p,q):
    '''
    Quaternion multiplication: Calculates the product of two quaternions r = p * q
    If one of both of the quaterions have only three columns,
    the scalar component is calculated such that the length
    of the quaternion is one.
    The lengths of the quaternions have to match, or one of
    the two quaternions has to have the length one.
    If both p and q only have 3 components, the returned quaternion
    also only has 3 components (i.e. the quaternion vector)
    
    Parameters
    ----------
    p,q : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors
    
    Returns
    -------
    r : quaternion or quaternion vector (if both
        p and q are contain quaternion vectors).
    
    Notes
    -----

    .. math::
        q \\circ p = \\sum\\limits_{i=0}^3 {q_i I_i} * \\sum\\limits_{j=0}^3 \\
        {p_j I_j} = (q_0 p_0 - \\vec{q} \\cdot \\vec{p}) + (q_0 \\vec{p} + p_0 \\
        \\vec{q} + \\vec{q} \\times \\vec{p}) \\cdot \\vec{I}
    
    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> p = [cos(0.2), 0, 0, sin(0.2)]
    >>> q = [[0, 0, 0.1],
    >>>    [0, 0.1, 0]]
    >>> r = quat.quatmult(p,q)

    '''

    flag3D = False
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape[1]==3 & q.shape[1]==3:
        flag3D = True

    if len(p) != len(q):
        assert (len(p)==1 or len(q)==1), \
            'Both arguments in the quaternion multiplication must have the same number of rows, unless one has only one row.'

    p = vect2quat(p).T
    q = vect2quat(q).T
    
    if np.prod(np.shape(p)) > np.prod(np.shape(q)):
        r=np.zeros(np.shape(p))
    else:
        r=np.zeros(np.shape(q))

    r[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    r[1] = p[1]*q[0] + p[0]*q[1] + p[2]*q[3] - p[3]*q[2]
    r[2] = p[2]*q[0] + p[0]*q[2] + p[3]*q[1] - p[1]*q[3]
    r[3] = p[3]*q[0] + p[0]*q[3] + p[1]*q[2] - p[2]*q[1]

    if flag3D:
        # for rotations > 180 deg
        r[:,r[0]<0] = -r[:,r[0]<0]
        r = r[1:]

    r = r.T
    return r

def quat2deg(inQuat):
    '''Calculate the axis-angle corresponding to a given quaternion.
    
    Parameters
    ----------
    inQuat: float, or array_like, shape ([3/4],) or (N,[3/4])
        quaternion(s) or quaternion vector(s)
    
    Returns
    -------
    axAng : corresponding axis angle(s)
        float, or shape (3,) or (N,3)
    
    Notes
    -----

    .. math::
        | \\vec{q} | = sin(\\theta/2)

    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> quat.quat2deg(0.1)
    array([ 11.47834095])

    >>> quat.quat2deg([0.1, 0.1, 0])
    array([ 11.47834095,  11.47834095,   0.        ])

    >>> quat.quat2deg([cos(0.1), 0, sin(0.1), 0])
    array([  0.       ,  11.4591559,   0.       ])
    '''
    return 2 * np.arcsin(quat2vect(inQuat)) * 180 / pi

def quat2rotmat(inQuat):
    ''' Calculate the rotation matrix corresponding to the quaternion. If
    "inQuat" contains more than one quaternion, the matrix is flattened (to
    facilitate the work with rows of quaternions), and can be restored to
    matrix form by "reshaping" the resulting rows into a (3,3) shape.
    
    Parameters
    ----------
    inQuat : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors
    
    Returns
    -------
    rotMat : corresponding rotation matrix/matrices (flattened)
    
    Notes
    -----

    .. math::
        {\\bf{R}} = \\left( {\\begin{array}{*{20}{c}}
        {q_0^2 + q_1^2 - q_2^2 - q_3^2}&{2({q_1}{q_2} - {q_0}{q_3})}&{2({q_1}{q_3} + {q_0}{q_2})}\\\\
        {2({q_1}{q_2} + {q_0}{q_3})}&{q_0^2 - q_1^2 + q_2^2 - q_3^2}&{2({q_2}{q_3} - {q_0}{q_1})}\\\\
        {2({q_1}{q_3} - {q_0}{q_2})}&{2({q_2}{q_3} + {q_0}{q_1})}&{q_0^2 - q_1^2 - q_2^2 + q_3^2} \\\\
        \\end{array}} \\right)

    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> r = quat.quat2rotmat([0, 0, 0.1])
    >>> r.shape
    (1, 9)
    >>> r.reshape((3,3))
    array([[ 0.98      , -0.19899749,  0.        ],
        [ 0.19899749,  0.98      ,  0.        ],
        [ 0.        ,  0.        ,  1.        ]])
    '''
    
    q = vect2quat(inQuat).T
    
    R = np.zeros((9, q.shape[1]))
    R[0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[2] = 2*(q[1]*q[3] + q[0]*q[2])
    R[3] = 2*(q[1]*q[2] + q[0]*q[3])
    R[4] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[5] = 2*(q[2]*q[3] - q[0]*q[1])
    R[6] = 2*(q[1]*q[3] - q[0]*q[2])
    R[7] = 2*(q[2]*q[3] + q[0]*q[1])
    R[8] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    
    if R.shape[1] == 1:
        return np.reshape(R, (3,3))
    else:
        return R.T
    
def quat2vect(inQuat):
    '''
    Extract the quaternion vector from a full quaternion.

    Parameters
    ----------
    inQuat : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors.
    
    Returns
    -------
    vect : array, shape (3,) or (N,3)
        corresponding quaternion vectors
    
    Notes
    -----
    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> quat.quat2vect([[cos(0.2), 0, 0, sin(0.2)],[cos(0.1), 0, sin(0.1), 0]])
    array([[ 0.        ,  0.        ,  0.19866933],
           [ 0.        ,  0.09983342,  0.        ]])

    '''
    
    inQuat = np.atleast_2d(inQuat)
    if inQuat.shape[1] == 4:
        vect = inQuat[:,1:]
    else:
        vect = inQuat
    if np.min(vect.shape)==1:
        vect = vect.ravel()
    return vect

def rotmat2quat(rMat):
    '''
    Assumes that R has the shape (3,3), or the matrix elements in columns

    Parameters
    ----------
    rMat : array, shape (3,3) or (N,9)
        single rotation matrix, or matrix with rotation-matrix elements.
    
    Returns
    -------
    outQuat : array, shape (4,) or (N,4)
        corresponding quaternion vector(s)
    
    Notes
    -----

    .. math::
         \\vec q = 0.5*copysign\\left( {\\begin{array}{*{20}{c}}
        {\\sqrt {1 + {R_{11}} - {R_{22}} - {R_{33}}} ,}\\\\
        {\\sqrt {1 - {R_{11}} + {R_{22}} - {R_{33}}} ,}\\\\
        {\\sqrt {1 - {R_{11}} - {R_{22}} + {R_{33}}} ,}
        \\end{array}\\begin{array}{*{20}{c}}
        {{R_{32}} - {R_{23}}}\\\\
        {{R_{13}} - {R_{31}}}\\\\
        {{R_{21}} - {R_{12}}}
        \\end{array}} \\right) 
    
    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    
    >>> rotMat = array([[cos(alpha), -sin(alpha), 0],
    >>>    [sin(alpha), cos(alpha), 0],
    >>>    [0, 0, 1]])
    >>> quat.rotmat2quat(rotMat)
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342]])
    
    '''    
    
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
    
def vect2quat(inData):
    ''' Utility function, which turns a quaternion vector into a unit quaternion.

    Parameters
    ----------
    inData : array_like, shape (3,) or (N,3)
        quaternions or quaternion vectors
    
    Returns
    -------
    quats : array, shape (4,) or (N,4)
        corresponding unit quaternions.
    
    Notes
    -----
    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> quats = array([[0,0, sin(0.1)],[0, sin(0.2), 0]])
    >>> quat.vect2quat(quats)
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342],
           [ 0.98006658,  0.        ,  0.19866933,  0.        ]])

    '''
    inData = np.atleast_2d(inData)
    (m,n) = inData.shape
    if (n!=3)&(n!=4):
        raise ValueError('Quaternion must have 3 or 4 columns')
    if n == 3:
        qLength = 1-np.sum(inData**2,1)
        numLimit = 1e-12
        # Check for numerical problems
        if np.min(qLength) < -numLimit:
            raise ValueError('Quaternion is too long!')
        else:
            # Correct for numerical problems
            qLength[qLength<0] = 0
        outData = np.hstack((np.c_[np.sqrt(qLength)], inData))
        
    else:
        outData = inData
        
    return outData

def vel2quat(omega, q0, rate, CStype):
    '''
    Take an angular velocity (in rad/s), and convert it into the
    corresponding orientation quaternion.

    Parameters
    ----------
    omega : array, shape (3,) or (N,3)
        angular velocity [rad/s].
    q0 : array (3,)
        vector-part of quaternion (!!)
    rate : float
        sampling rate (in [Hz])
    CStype:  string
        coordinate_system, space-fixed ("sf") or body_fixed ("bf")

    Returns
    -------
    quats : array, shape (N,4)
        unit quaternion vectors.

    Notes
    -----
    1) The output has the same length as the input. As a consequence, the last velocity vector is ignored.
    2) For angular velocity with respect to space ("sf"), the orientation is given by

      .. math::
          q(t) = \\Delta q(t_n) \\circ \\Delta q(t_{n-1}) \\circ ... \\circ \\Delta q(t_2) \\circ \\Delta q(t_1) \\circ q(t_0)

      .. math::
        \\Delta \\vec{q_i} = \\vec{n(t)}\\sin (\\frac{\\Delta \\phi (t_i)}{2}) = \\frac{\\vec \\omega (t_i)}{\\left| {\\vec \\omega (t_i)} \\right|}\\sin \\left( \\frac{\\left| {\\vec \\omega ({t_i})} \\right|\\Delta t}{2} \\right)

    3) For angular velocity with respect to the body ("bf"), the sequence of quaternions is inverted.

    4) Take care that you choose a high enough sampling rate!

    Examples
    --------
    >>> v0 = np.r_[0., 0., 100.] * np.pi/180.
    >>> omega = np.tile(v0, (1000,1))
    >>> rate = 100
    >>> out = quat.vel2quat(omega, [0., 0., 0.], rate, 'sf')
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.99996192,  0.        ,  0.        ,  0.00872654],
       [ 0.9998477 ,  0.        ,  0.        ,  0.01745241],
       ..., 
       [-0.74895572,  0.        ,  0.        ,  0.66262005],
       [-0.75470958,  0.        ,  0.        ,  0.65605903],
       [-0.76040597,  0.        ,  0.        ,  0.64944805]])
    '''
    
    omega = np.atleast_2d(omega)
    
    # The following is (approximately) the quaternion-equivalent of the trapezoidal integration (cumtrapz)
    if omega.shape[1]>1:
        omega[:-1] = 0.5*(omega[:-1] + omega[1:])

    omega_t = np.sqrt(np.sum(omega**2, 1))
    omega_nonZero = omega_t>0

    # initialize the quaternion
    q_delta = np.zeros(omega.shape)
    q_pos = np.zeros((len(omega),4))
    q_pos[0,:] = vect2quat(q0)

    # magnitude of position steps
    dq_total = np.sin(omega_t[omega_nonZero]/(2.*rate))

    q_delta[omega_nonZero,:] = omega[omega_nonZero,:] * np.tile(dq_total/omega_t[omega_nonZero], (3,1)).T

    for ii in range(len(omega)-1):
        q1 = vect2quat(q_delta[ii,:])
        q2 = q_pos[ii,:]
        if CStype == 'sf':            
            qm = quatmult(q1,q2)
        elif CStype == 'bf':
            qm = quatmult(q2,q1)
        else:
            print('I don''t know this type of coordinate system!')
        q_pos[ii+1,:] = qm

    return q_pos

def quat2vel(q, rate=1, winSize=5, order=2):
    '''
    Take a quaternion, and convert it into the
    corresponding angular velocity

    Parameters
    ----------
    q : array, shape (N,[3,4])
        unit quaternion vectors.
    rate : float
        sampling rate (in [Hz])
    winSize : integer
        window size for the calculation of the velocity.
        Has to be odd.
    order : integer
        Order of polynomial used by savgol to calculate the first derivative

    Returns
    -------
    vel : array, shape (3,) or (N,3)
        angular velocity [rad/s].

    Notes
    -----
    The angular velocity is given by

      .. math::
        \\omega = 2 * \\frac{dq}{dt} \\circ q^{-1}

    Examples
    --------
    >>> rate = 1000
    >>> t = np.arange(0,10,1/rate)
    >>> x = 0.1 * np.sin(t)
    >>> y = 0.2 * np.sin(t)
    >>> z = np.zeros_like(t)
    array([[ 0.20000029,  0.40000057,  0.        ],
           [ 0.19999989,  0.39999978,  0.        ],
           [ 0.19999951,  0.39999901,  0.        ]])
            .......
    '''
    
    if np.mod(winSize, 2) != 1:
        raise ValueError('Window size must be odd!')
    
    numCols = q.shape[1]
    if numCols < 3 or numCols > 4:
        raise TypeError('quaternions must have 3 or 4 columns')
    
    # This has to be done: otherwise quatmult will "complete" dq_dt to be a unit
    # quaternion, resulting in wrong value
    if numCols == 3:
        q = vect2quat(q)
    
    dq_dt = signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=1, delta=1./rate, axis=0)
    angVel = 2 * quatmult(dq_dt, quatinv(q))
    
    return angVel[:,1:]
    
if __name__=='__main__':
    '''These are some simple tests to see if the functions produce the
    proper output.
    More extensive tests are found in tests/test_quat.py'''
    
    from skinematics.vector import rotate_vector
    
    v0 = np.r_[0., 0., 100.] * np.pi/180.
    vel = np.tile(v0, (1000,1))
    rate = 100
    out = vel2quat(vel, [0., 0., 0.], rate, 'sf')
    
    rate = 1000
    t = np.arange(0,10,1./rate)
    x = 0.1 * np.sin(t)
    y = 0.2 * np.sin(t)
    z = np.zeros_like(t)
    
    q = np.column_stack( (x,y,z) )
    vel = quat2vel(q, rate, 5, 2)
    qReturn = vel2quat(vel, q[0], rate, 'sf' )
    plt.plot(q)
    plt.plot(qReturn[:,1:],'--')
    plt.show()
    
    q = Quaternion(np.array([0,0,10]), 'Fick')
    print(q)
    rm = q.export(to='rotmat')
    print(rm)
    q2 = Quaternion(rm, inType='rotmat') 
    print(q2)
    
        
    a = np.r_[np.cos(0.1), 0,0,np.sin(0.1)]
    b = np.r_[np.cos(0.1),0,np.sin(0.1), 0]
    c = np.vstack((a,b))
    d = np.r_[np.sin(0.1), 0, 0]
    e = np.r_[2, 0, np.sin(0.1), 0]

    print(quatmult(a,a))
    print(quatmult(a,b))
    print(quatmult(c,c))
    print(quatmult(c,a))
    print(quatmult(d,d))

    print('The inverse of {0} is {1}'.format(a, quatinv(a)))
    print('The inverse of {0} is {1}'.format(d, quatinv(d)))
    print('The inverse of {0} is {1}'.format(e, quatinv(e)))
    print(quatmult(e, quatinv(e)))

    print(quat2vect(a))
    print('{0} is {1} degree'.format(a, quat2deg(a)))
    print('{0} is {1} degree'.format(c, quat2deg(c)))
    print(quat2deg(0.2))
    x = np.r_[1,0,0]
    vNull = np.r_[0,0,0]
    print(rotate_vector(x, a))

    v0 = [0., 0., 100.]
    vel = np.tile(v0, (1000,1))
    rate = 100
    
    out = vel2quat(vel, [0., 0., 0.], rate, 'sf')
    print(out[-1:])
    plt.plot(out[:,1:4])
    plt.show()
    
    print(deg2quat(15))
    print(deg2quat(quat2deg(a)))
    
    q = np.array([[0, 0, np.sin(0.1)],
               [0, np.sin(0.01), 0]])
    rMat = quat2rotmat(q)
    print(rMat[1].reshape((3,3)))
    qNew = rotmat2quat(rMat)
    print(qNew)
    
    q = Quaternion(np.array([0,0,0.5]))
    p = Quaternion(np.array([[0,0,0.5], [0,0,0.1]]))
    r = Quaternion([0,0,0.5])
    print(p*q)
    print(q*3)
    print(q*pi)
    print(q/p)
    print(q/5)
    
    Fick = p.export('Fick')
    Q_fick = q.export('Fick')
    
    import pprint
    pprint.pprint(Fick)
    pprint.pprint(np.rad2deg(Fick))
    
    p = Quaternion(np.array([[0,0,0.5], [0,0,0.1], [0,0,0.1]]))
    p[1:] = [[0,0,0],[0,0,0.01]]
    print(p)
    
