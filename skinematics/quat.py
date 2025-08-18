"""
Functions for working with quaternions. Note that all the functions also
work on arrays, and can deal with full quaternions as well as with
quaternion vectors.
"""

# author: Thomas Haslwanter
# date:   Feb-2024

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys

file_dir = os.path.dirname(__file__)
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

import vector, rotmat

#import deprecation
#import warnings
#warnings.simplefilter('always', DeprecationWarning)

pi = np.pi

class Quaternion():
    """
    Quaternion class, with multiplication, division, and inversion.
    A Quaternion can be created from vectors, rotation matrices,
    or from Fick-angles, Helmholtz-angles, or Euler angles ( in deg).
    It provides

    * operator overloading for mult, div, and inv.
    * indexing
    * access to the data, in the attribute * values * .

    Parameters
    ----------
    inData: np.ndarray
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


    Attribute
    ---------
    values : (4 x n) array
        quaternion values

    Method
    ------
    inv() : Inverse of the quaterion

    export(to='rotmat') : Export to one of the following formats: 'rotmat', 'Euler', 'Fick', 'Helmholtz'


    Note
    ----
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

    >>> q = Quaternion(array([[0,0,0.1], [0,0,0.2], [0,0,0.5]]))
    >>> p = Quaternion(array([0,0,0.2]))
    >>> fick = Quaternion( array([[0,0,10], [0,10,10]]), 'Fick')
    >>> combined = p * q
    >>> divided = q / p
    >>> extracted = q[1:2]
    >>> len(q)
    >>> data = q.values
    >>> 2
    >>> inv(q)

    """

    def __init__(self, inData, inType='vector'):
        """ Initialization """

        if inType.lower() == 'vector':
            if isinstance(inData, np.ndarray) or isinstance(inData, list):
                self.values = unit_q(inData)
            elif isinstance(inData, Quaternion):
                self.values = inData.values
            else:
                raise TypeError('Quaternions can only be based on ndarray or Quaternions!')

        elif inType.lower() == 'rotmat':
            """Conversion from rotation matrices to quaternions."""
            self.values = rotmat2quat(inData)

        elif inType.lower() == 'euler':
            """ Conversion from Euler angles to quaternions.
            (a,b,g) stands for (alpha, beta, gamma) """

            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)

            (ca, cb, cg) = np.cos(inData.T)
            (sa, sb, sg) = np.sin(inData.T)

            self.values = np.vstack( (ca*cb*cg - sa*cb*sg,
                                      ca*sb*cg + sa*sb*sg,
                                      ca*sb*sg - sa*sb*cg,
                                      ca*cb*sg + sa*cb*cg) ).T
        elif inType.lower() == 'fick':
            """ Conversion from Fick angles to quaternions.
            (p,f,t) stands for (psi, phi, theta) """

            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)

            (cp, cf, ct) = np.cos(inData.T)
            (sp, sf, st) = np.sin(inData.T)

            self.values = np.vstack( (cp*cf*ct + sp*sf*st,
                                      sp*cf*ct - cp*sf*st,
                                      cp*sf*ct + sp*cf*st,
                                      cp*cf*st - sp*sf*ct) ).T

        elif inType.lower() == 'helmholtz':
            """ Conversion from Helmholtz angles to quaternions.
            (p,f,t) stands for (psi, phi, theta) """

            inData[inData<0] += 360
            inData = np.deg2rad(inData/2)

            (cp, cf, ct) = np.cos(inData.T)
            (sp, sf, st) = np.sin(inData.T)

            self.values = np.vstack( (cp*cf*ct - sp*sf*st,
                                      sp*cf*ct + cp*sf*st,
                                      cp*sf*ct + sp*cf*st,
                                      cp*cf*st - sp*sf*ct ) ).T

    def __len__(self):
        """The "length" is given by the number of quaternions."""
        return len(self.values)

    def __mul__(self, other):
        """Operator overloading for multiplication."""
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values * other)
        else:
            return Quaternion(q_mult(self.values, other.values))

    def __div__(self, other):
        """Operator overloading for division."""
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values / other)
        else:
            return Quaternion(q_mult(self.values, q_inv(other.values)))

    def __truediv__(self, other):
        """Operator overloading for division."""
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self.values / other)
        else:
            return Quaternion(q_mult(self.values, q_inv(other.values)))

    def __getitem__(self, select):
        return Quaternion(self.values[select])

    def __setitem__(self, select, item):
        self.values[select] = unit_q(item)

    #def __delitem__(self, select):
        #np.delete(self.values, select, axis=0)

    def inv(self):
        """Inverse of a quaternion."""
        return Quaternion(q_inv(self.values))

    def __repr__(self):
        return 'Quaternion ' + str(self.values)

    def export(self, to='rotmat'):
        """
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

        """
        if to.lower() == 'rotmat' :
           return convert(self.values, 'rotmat')

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

def convert(quat, to='rotmat'):
    """ Calculate the rotation matrix corresponding to the quaternion. If
    "inQuat" contains more than one quaternion, the matrix is flattened (to
    facilitate the work with rows of quaternions), and can be restored to
    matrix form by "reshaping" the resulting rows into a (3,3) shape.

    Parameters
    ----------
    inQuat : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors
    to : string
        Has to be one of the following:

        - rotmat : rotation matrix
        - Gibbs  : Gibbs vector

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
    >>> r = quat.convert([0, 0, 0.1], to='rotmat')
    >>> r.shape
    (1, 9)
    >>> r.reshape((3,3))
    array([[ 0.98      , -0.19899749,  0.        ],
        [ 0.19899749,  0.98      ,  0.        ],
        [ 0.        ,  0.        ,  1.        ]])
    """

    if to == 'rotmat':
        q = unit_q(quat).T

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

    elif to == 'Gibbs':
        q_0 = q_scalar(quat)      # cos(alpha/2)
        gibbs = (q_vector(quat).T / q_0).T    # tan = sin/cos

        return gibbs


def deg2quat(inDeg):
    """
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

    """
    deg = (inDeg+180)%360-180
    return np.sin(0.5 * deg * pi/180)

def q_conj(q):
    """ Conjugate quaternion

    Parameters
    ----------
    q: array_like, shape ([3,4],) or (N,[3/4])
        quaternion or quaternion vectors

    Returns
    -------
    qconj : conjugate quaternion(s)


    Examples
    --------
    >>>  quat.q_conj([0,0,0.1])
    array([ 0.99498744, -0.        , -0.        , -0.1       ])

    >>> quat.q_conj([[cos(0.1),0,0,sin(0.1)],
    >>>    [cos(0.2), 0, sin(0.2), 0]])
    array([[ 0.99500417, -0.        , -0.        , -0.09983342],
           [ 0.98006658, -0.        , -0.19866933, -0.        ]])

    """

    q = np.atleast_2d(q)
    if q.shape[1]==3:
        q = unit_q(q)

    qConj = q * np.r_[1, -1,-1,-1]

    if q.shape[0]==1:
        qConj=qConj.ravel()

    return qConj


def q_inv(q):
    """ Quaternion inversion

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
    >>>  quat.q_inv([0,0,0.1])
    array([-0., -0., -0.1])

    >>> quat.q_inv([[cos(0.1),0,0,sin(0.1)],
    >>> [cos(0.2),0,sin(0.2),0]])
    array([[ 0.99500417, -0.        , -0.        , -0.09983342],
           [ 0.98006658, -0.        , -0.19866933, -0.        ]])
    """

    q = np.atleast_2d(q)
    if q.shape[1]==3:
        return -q
    else:
        qLength = np.sum(q**2, 1)
        qConj = q * np.r_[1, -1,-1,-1]
        return (qConj.T / qLength).T


def q_mult(p,q):
    """
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
    >>> r = quat.q_mult(p,q)

    """

    flag3D = False
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape[1]==3 & q.shape[1]==3:
        flag3D = True

    if len(p) != len(q):
        assert (len(p)==1 or len(q)==1), \
            'Both arguments in the quaternion multiplication must have the same number of rows, unless one has only one row.'

    p = unit_q(p).T
    q = unit_q(q).T

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
    """Calculate the axis-angle corresponding to a given quaternion.

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
    """
    return 2 * np.arcsin(q_vector(inQuat)) * 180 / pi



def quat2seq(quats, seq='nautical'):
    """
    This function takes a quaternion, and calculates the corresponding
    angles for sequenctial rotations.

    Parameters
    ----------
    quats : ndarray, nx4
        input quaternions
    seq : string
        Has to be one the following:

        - Euler ... Rz * Rx * Rz
        - Fick ... Rz * Ry * Rx
        - nautical ... same as "Fick"
        - Helmholtz ... Ry * Rz * Rx

    Returns
    -------
    sequence : ndarray, nx3
        corresponding angles [deg]
        same sequence as in the rotation matrices

    Examples
    --------
    >>> quat.quat2seq([0,0,0.1])
    array([[ 11.47834095,  -0.        ,   0.        ]])

    >>> quaternions = [[0,0,0.1], [0,0.2,0]]
    skin.quat.quat2seq(quaternions)
    array([[ 11.47834095,  -0.        ,   0.        ],
           [  0.        ,  23.07391807,   0.        ]])

    >>> skin.quat.quat2seq(quaternions, 'nautical')
    array([[ 11.47834095,  -0.        ,   0.        ],
           [  0.        ,  23.07391807,   0.        ]])

    >>> skin.quat.quat2seq(quaternions, 'Euler')
    array([[ 11.47834095,   0.        ,   0.        ],
           [ 90.        ,  23.07391807,  -90.        ]])


    """

    # Ensure that it also works for a single quaternion
    quats = np.atleast_2d(quats)

    # If only the quaternion vector is entered, extend it to a full unit quaternion
    if quats.shape[1] == 3:
        quats = unit_q(quats)

    if seq =='Fick' or seq =='nautical':
        R_zx = 2 * (quats[:,1]*quats[:,3] - quats[:,0]*quats[:,2])
        R_yx = 2 * (quats[:,1]*quats[:,2] + quats[:,0]*quats[:,3])
        R_zy = 2 * (quats[:,2]*quats[:,3] + quats[:,0]*quats[:,1])

        phi  = -np.arcsin(R_zx)
        theta = np.arcsin(R_yx / np.cos(phi))
        psi   = np.arcsin(R_zy / np.cos(phi))

        sequence = np.column_stack((theta, phi, psi))

    elif seq == 'Helmholtz':
        R_yx = 2 * (quats[:,1]*quats[:,2] + quats[:,0]*quats[:,3])
        R_zx = 2 * (quats[:,1]*quats[:,3] - quats[:,0]*quats[:,2])
        R_yz = 2 * (quats[:,2]*quats[:,3] - quats[:,0]*quats[:,1])

        theta = np.arcsin(R_yx)
        phi  = -np.arcsin(R_zx / np.cos(theta))
        psi  = -np.arcsin(R_yz / np.cos(theta))

        sequence = np.column_stack((phi, theta, psi))

    elif seq == 'Euler':
        Rs = convert(quats, to='rotmat').reshape((-1,3,3))

        beta = np.arccos(Rs[:,2,2])

        # special handling for (beta == 0)
        bz = beta == 0

        # there the gamma-values are set to zero, since alpha/gamma is degenerated
        alpha = np.nan * np.ones_like(beta)
        gamma = np.nan * np.ones_like(beta)

        alpha[bz] = np.arcsin(Rs[bz,1,0])
        gamma[bz] = 0

        alpha[~bz] = np.arctan2(Rs[~bz,0,2], Rs[~bz,1,2])
        gamma[~bz] = np.arctan2(Rs[~bz,2,0], Rs[~bz,2,1])

        sequence = np.column_stack((alpha, beta, gamma))
    else:
        raise ValueError('Input parameter {0} not known'.format(seq))

    return np.rad2deg(sequence)

def q_vector(inQuat):
    """
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
    >>> quat.q_vector([[np.cos(0.2), 0, 0, np.sin(0.2)],[cos(0.1), 0, np.sin(0.1), 0]])
    array([[ 0.        ,  0.        ,  0.19866933],
           [ 0.        ,  0.09983342,  0.        ]])

    """

    inQuat = np.atleast_2d(inQuat)
    if inQuat.shape[1] == 4:
        vect = inQuat[:,1:]
    else:
        vect = inQuat
    if np.min(vect.shape)==1:
        vect = vect.ravel()
    return vect

def q_scalar(inQuat):
    """
    Extract the quaternion scalar from a full quaternion.

    Parameters
    ----------
    inQuat : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors.

    Returns
    -------
    vect : array, shape (1,) or (N,1)
        Corresponding quaternion scalar.
        If the input is only the quaternion-vector, the scalar part for a unit
        quaternion is calculated and returned.

    Notes
    -----
    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------
    >>> quat.q_scalar([[np.cos(0.2), 0, 0, np.sin(0.2)],[np.cos(0.1), 0, np.sin(0.1), 0]])
    array([ 0.98006658,  0.99500417])

    """

    inQuat = np.atleast_2d(inQuat)
    if inQuat.shape[1] == 4:
        scalar = inQuat[:,0]
    else:
        scalar = np.sqrt(1-np.linalg.norm(inQuat, axis=1))
    if np.min(scalar.shape)==1:
        scalar = scalar.ravel()
    return scalar


def unit_q(inData):
    """ Utility function, which turns a quaternion vector into a unit quaternion.
    If the input is already a full quaternion, the output equals the input.

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
    >>> quat.unit_q(quats)
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342],
           [ 0.98006658,  0.        ,  0.19866933,  0.        ]])

    """
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

def calc_quat(omega, q0, rate, CStype):
    """
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
    >>> out = quat.calc_quat(omega, [0., 0., 0.], rate, 'sf')
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.99996192,  0.        ,  0.        ,  0.00872654],
       [ 0.9998477 ,  0.        ,  0.        ,  0.01745241],
       ...,
       [-0.74895572,  0.        ,  0.        ,  0.66262005],
       [-0.75470958,  0.        ,  0.        ,  0.65605903],
       [-0.76040597,  0.        ,  0.        ,  0.64944805]])
    """

    omega_05 = np.atleast_2d(omega).copy()

    # The following is (approximately) the quaternion-equivalent of the trapezoidal integration (cumtrapz)
    if omega_05.shape[1]>1:
        omega_05[:-1] = 0.5*(omega_05[:-1] + omega_05[1:])

    omega_t = np.sqrt(np.sum(omega_05**2, 1))
    omega_nonZero = omega_t>0

    # initialize the quaternion
    q_delta = np.zeros(omega_05.shape)
    q_pos = np.zeros((len(omega_05),4))
    q_pos[0,:] = unit_q(q0)

    # magnitude of position steps
    dq_total = np.sin(omega_t[omega_nonZero]/(2.*rate))

    q_delta[omega_nonZero,:] = omega_05[omega_nonZero,:] * np.tile(dq_total/omega_t[omega_nonZero], (3,1)).T

    for ii in range(len(omega_05)-1):
        q1 = unit_q(q_delta[ii,:])
        q2 = q_pos[ii,:]
        if CStype == 'sf':
            qm = q_mult(q1,q2)
        elif CStype == 'bf':
            qm = q_mult(q2,q1)
        else:
            print('I don''t know this type of coordinate system!')
        q_pos[ii+1,:] = qm

    return q_pos


def calc_angvel(q, rate=1, winSize=5, order=2):
    """
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
    angvel : array, shape (3,) or (N,3)
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
    """

    if np.mod(winSize, 2) != 1:
        raise ValueError('Window size must be odd!')

    numCols = q.shape[1]
    if numCols < 3 or numCols > 4:
        raise TypeError('quaternions must have 3 or 4 columns')

    # This has to be done: otherwise q_mult will "complete" dq_dt to be a unit
    # quaternion, resulting in wrong value
    if numCols == 3:
        q = unit_q(q)

    dq_dt = signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=1, delta=1./rate, axis=0)
    angVel = 2 * q_mult(dq_dt, q_inv(q))

    return angVel[:,1:]

if __name__=='__main__':
    """These are some simple tests to see if the functions produce the
    proper output.
    More extensive tests are found in tests/test_quat.py"""

    a = np.r_[np.cos(0.1), 0,0,np.sin(0.1)]
    b = np.r_[np.cos(0.2), 0,np.sin(0.2),0]
    seq = quat2seq(np.vstack((a,b)), seq='Euler')
    print(seq)

    """
    from skinematics.vector import rotate_vector

    v0 = np.r_[0., 0., 100.] * np.pi/180.
    vel = np.tile(v0, (1000,1))
    rate = 100
    out = calc_quat(vel, [0., 0., 0.], rate, 'sf')

    rate = 1000
    t = np.arange(0,10,1./rate)
    x = 0.1 * np.sin(t)
    y = 0.2 * np.sin(t)
    z = np.zeros_like(t)

    q = np.column_stack( (x,y,z) )
    vel = calc_angvel(q, rate, 5, 2)
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

    print(q_mult(a,a))
    print(q_mult(a,b))
    print(q_mult(c,c))
    print(q_mult(c,a))
    print(q_mult(d,d))

    print('The inverse of {0} is {1}'.format(a, q_inv(a)))
    print('The inverse of {0} is {1}'.format(d, q_inv(d)))
    print('The inverse of {0} is {1}'.format(e, q_inv(e)))
    print(q_mult(e, q_inv(e)))

    print(q_vector(a))
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
    rMat = convert(q, to='rotmat)
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

    """
