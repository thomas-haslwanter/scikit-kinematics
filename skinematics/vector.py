'''
Routines for working with vectors
These routines can be used with vectors, as well as with matrices containing a vector in each row.
'''
 
'''
author :  Thomas Haslwanter
date :    July-2017
'''

import numpy as np


# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) ) 

#PACKAGE_PARENT = '..'
#SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
#sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from skinematics import quat

# For deprecation warnings
#import deprecation
import warnings
#warnings.simplefilter('always', DeprecationWarning)


def normalize(v):
    ''' Normalization of a given vector (with image)
    
    Parameters
    ----------
    v : array (N,) or (M,N)
        input vector

    Returns
    -------
    v_normalized : array (N,) or (M,N)
        normalized input vector


    .. image:: ../docs/Images/vector_normalize.png
        :scale: 33%

    Example
    -------

    >>> skinematics.vector.normalize([3, 0, 0])
    array([[ 1.,  0.,  0.]])
    
    >>> v = [[pi, 2, 3], [2, 0, 0]]
    >>> skinematics.vector.normalize(v)
    array([[ 0.6569322 ,  0.41821602,  0.62732404],
       [ 1.        ,  0.        ,  0.        ]])
    
    Notes
    -----

    .. math::
        \\vec{n} = \\frac{\\vec{v}}{|\\vec{v}|}

        

    '''
    
    from numpy.linalg import norm
    
    if np.array(v).ndim == 1:
        vectorFlag = True
    else:
        vectorFlag = False
        
    v = np.atleast_2d(v)
    length = norm(v,axis=1)
    if vectorFlag:
        v = v.ravel()
    return (v.T/length).T


def angle(v1,v2):
    '''Angle between two vectors
    
    Parameters
    ----------
    v1 : array (N,) or (M,N)
        vector 1
    v2 : array (N,) or (M,N)
        vector 2

    Returns
    -------
    angle : double or array(M,)
        angle between v1 and v2


    .. image:: ../docs/Images/vector_angle.png
        :scale: 33%

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    >>>       [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    >>>       [0,1,0]])
    >>> skinematics.vector.angle(v1,v2)
    array([ 1.30024656,  0.96453036])
    
    Notes
    -----

    .. math::
        \\alpha =arccos(\\frac{\\vec{v_1} \\cdot \\vec{v_2}}{| \\vec{v_1} |
        \\cdot | \\vec{v_2}|})


    '''

    
    # make sure lists are handled correctly
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if v1.ndim < v2.ndim:
        v1, v2 = v2, v1
    n1 = normalize(v1)
    n2 = normalize(v2)
    if v2.ndim == 1:
        angle = np.arccos(n1.dot(n2))
    else:
        angle = np.arccos(list(map(np.dot, n1, n2)))
    return angle
 

def project(v1,v2, projection_type='1D'):
    '''Project one vector onto another, or into the plane perpendicular to that vector.
    
    Parameters
    ----------
    v1 : array (N,) or (M,N)
        projected vector
    v2 : array (N,) or (M,N):
        target vector
    projection_type : scalar
        Has to be one of the following:
        
        - 1D ... projection onto a vector (Default)
        - 2D ... projection into the plane perpendicular to that vector


    Returns
    -------
    v_projected : array (N,) or (M,N)
        projection of v1 onto v2


    .. image:: ../docs/Images/vector_project.png
        :scale: 33%

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    >>>       [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    >>>       [0,1,0]])
    >>> skinematics.vector.project(v1,v2)
    array([[ 1.,  0.,  0.],
       [ 0.,  5.,  0.]])
     
    Notes
    -----

    .. math::
        \\vec{n} = \\frac{ \\vec{a} }{| \\vec{a} |}

        \\vec{v}_{proj} = \\vec{n} (\\vec{v} \\cdot \\vec{n})

        \\mathbf{c}^{image} = \mathbf{R} \cdot \mathbf{c}^{space} + \mathbf{p}_{CS}

    *Note* that the orientation of the 2D projection is not uniquely defined.
    It is chosen here such that the y-axis points up, and one is "looking down"
    rather than "looking up".
    

    '''
    
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)
    
    e2 = normalize(v2)
    
    if projection_type == '1D':
        if e2.ndim == 1 or e2.shape[0]==1:
            return (e2 * list(map(np.dot, v1, e2))).ravel()
        else:
            return (e2.T * list(map(np.dot, v1, e2))).T
    elif projection_type == '2D':
        if e2.shape[0] > 1:
            raise ValueError('2D projections only implemented for fixed projection-plane!')
            
        x,y,z = e2[0]
        projection_matrix = np.array(
            [[-y,      -x*z, x],
             [ x,      -y*z, y],
             [ 0, x**2+y**2, z]])
        
        if z > 0:    # choose a downward-pointing look for the projection
            projection_matrix  = projection_matrix * np.r_[-1, 1, -1]
        
        projected = v1 @ projection_matrix
        projected = projected[:,:2]
        if e2.ndim == 1 or e2.shape[0]==1:
            return projected.ravel()
        else:
            return projected
    else:
        raise ValueError('{0} not allowed as projection_type in vector.project!'.format(projection_type))
        
 
def GramSchmidt(p0,p1,p2):
    '''Gram-Schmidt orthogonalization
    
    Parameters
    ----------
    p0 : array (3,) or (M,3)
        coordinates of Point 1
    p1 : array (3,) or (M,3)
        coordinates of Point 2
    p2 : array (3,) or (M,3)
        coordinates of Point 3

    Returns
    -------
    Rmat : array (9,) or (M,9)
        flattened rotation matrix


    .. image:: ../docs/Images/GramSchmidt.jpg
        :scale: 25%

    Example
    -------
    >>> P0 = np.array([[0, 0, 0], [1,2,3]])
    >>> P1 = np.array([[1, 0, 0], [4,1,0]])
    >>> P2 = np.array([[1, 1, 0], [9,-1,1]])
    >>> GramSchmidt(P0,P1,P2)
    array([[ 1.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  1.        ],
       [ 0.6882472 , -0.22941573, -0.6882472 ,  0.62872867, -0.28470732,
         0.72363112, -0.36196138, -0.93075784, -0.05170877]])

    Notes
    -----
    
    The flattened rotation matrix corresponds to 

    .. math::
        \\mathbf{R} = [ \\vec{e}_1 \\, \\vec{e}_2 \\, \\vec{e}_3 ]

    '''
    
    # If inputs are lists, convert them to arrays:
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    v1 = np.atleast_2d(p1-p0)
    v2 = np.atleast_2d(p2-p0)
        
    ex = normalize(v1)
    ey = normalize(v2- project(v2,ex))
    ez = np.cross(ex,ey)
    
    return np.hstack((ex,ey,ez))

def plane_orientation(p0, p1, p2):
    '''The vector perpendicular to the plane defined by three points.
    
    Parameters
    ----------
    p0 : array (3,) or (M,3)
        coordinates of Point 0
    p1 : array (3,) or (M,3)
        coordinates of Point 1
    p2 : array (3,) or (M,3)
        coordinates of Point 2

    Returns
    -------
    n : array (3,) or (M,3)
        vector perpendicular to the plane


    .. image:: ../docs/Images/vector_plane_orientation.png
        :scale: 33%

    Example
    -------
    >>> P0 = np.array([[0, 0, 0], [1,2,3]])
    >>> P1 = np.array([[1, 0, 0], [4,1,0]])
    >>> P2 = np.array([[1, 1, 0], [9,-1,1]])
    >>> plane_orientation(P0,P1,P2)
    array([[ 0.        ,  0.        ,  1.        ],
           [-0.36196138, -0.93075784, -0.05170877]])

    Notes
    -----

    .. math::
        \\vec{n} = \\frac{ \\vec{a} \\times \\vec{b}} {| \\vec{a} \\times \\vec{b}|}
    

    '''
    
    # If inputs are lists, convert them to arrays:
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)

    v01 = p1-p0
    v02 = p2-p0
    n = np.cross(v01,v02)
    return normalize(n)

#@deprecation.deprecated(deprecated_in="1.7", removed_in="1.9",
                        #current_version=__version__,
                        #details="Use the ``q_shortest_rotation`` function instead")
                        
def q_shortest_rotation(v1,v2):
    '''Quaternion indicating the shortest rotation from one vector into another.
    You can read "qrotate" as either "quaternion rotate" or as "quick
    rotate".
    
    Parameters
    ----------
    v1 : ndarray (3,)
        first vector
    v2 : ndarray (3,)
        second vector
        
    Returns
    -------
    q : ndarray (3,) 
        quaternion rotating v1 into v2
        

    .. image:: ../docs/Images/vector_q_shortest_rotation.png
        :scale: 33%

    Example
    -------
    >>> v1 = np.r_[1,0,0]
    >>> v2 = np.r_[1,1,0]
    >>> q = qrotate(v1, v2)
    >>> print(q)
    [ 0.          0.          0.38268343]
    '''
    
    # calculate the direction
    n = normalize(np.cross(v1,v2))
    
    # make sure vectors are handled correctly
    n = np.atleast_2d(n)
    
    # handle 0-quaternions
    nanindex = np.isnan(n[:,0])
    n[nanindex,:] = 0
    
    # find the angle, and calculate the quaternion
    angle12 = angle(v1,v2)
    q = (n.T*np.sin(angle12/2.)).T
    
    # if you are working with vectors, only return a vector
    if q.shape[0]==1:
        q = q.flatten()
        
    return q
    

def rotate_vector(vector, q):
    '''
    Rotates a vector, according to the given quaternions.
    Note that a single vector can be rotated into many orientations;
    or a row of vectors can all be rotated by a single quaternion.
    
    
    Parameters
    ----------
    vector : array, shape (3,) or (N,3)
        vector(s) to be rotated.
    q : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors.
    
    Returns
    -------
    rotated : array, shape (3,) or (N,3)
        rotated vector(s)
    

    .. image:: ../docs/Images/vector_rotate_vector.png
        :scale: 33%

    Notes
    -----
    .. math::
        q \\circ \\left( {\\vec x \\cdot \\vec I} \\right) \\circ {q^{ - 1}} = \\left( {{\\bf{R}} \\cdot \\vec x} \\right) \\cdot \\vec I

    More info under 
    http://en.wikipedia.org/wiki/Quaternion
    
    Examples
    --------
    >>> mymat = eye(3)
    >>> myVector = r_[1,0,0]
    >>> quats = array([[0,0, sin(0.1)],[0, sin(0.2), 0]])
    >>> quat.rotate_vector(myVector, quats)
    array([[ 0.98006658,  0.19866933,  0.        ],
           [ 0.92106099,  0.        , -0.38941834]])

    >>> quat.rotate_vector(mymat, [0, 0, sin(0.1)])
    array([[ 0.98006658,  0.19866933,  0.        ],
           [-0.19866933,  0.98006658,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    '''
    vector = np.atleast_2d(vector)
    qvector = np.hstack((np.zeros((vector.shape[0],1)), vector))
    vRotated = quat.q_mult(q, quat.q_mult(qvector, quat.q_inv(q)))
    vRotated = vRotated[:,1:]

    if min(vRotated.shape)==1:
        vRotated = vRotated.ravel()

    return vRotated


def target2orient(target, orient_type='quat'):
    ''' Converts a target vector into a corresponding orientation.
    Useful for targeting devices, such as eyes, cameras, or missile trackers.
    Based on the assumption, that in the reference orientation, the targeting
    device points forward.
    
    Parameters
    ----------
    target : array (3,) or (N,3)
        Input vector
    orient_type : string
        Has to be one the following:
        
        - Fick ... Rz * Ry
        - nautical ... same as "Fick"
        - Helmholtz ... Ry * Rz
        - quat ... quaternion

    Returns
    -------
    orientation : array (3,) or (N,3)
        Corresponding orientation
        For rotation matrices, same sequence as the matrices [deg].
        For quaternions, the quaternion vector.
        
        Note that the last column of the sequence angles, and the first column
        of the quaterion, will always be zero, because a rotation about
        the line-of-sight has no effect.

    Example
    -------

    >>> a = [3,3,0]
    >>> b = [5., 0, 5]
    >>> skinematics.vector.target2orient(a)
    [ 0.          0.          0.38268343]
    
    >>> skinematics.vector.target2orient([a,b])
    [[ 0.          0.          0.38268343]
     [ 0.         -0.38268343  0.        ]]
    
    >>> skinematics.vector.target2orient(a, orient_type='nautical')
    [ 45.  -0.   0.]
    '''
    
    if orient_type == 'quat':
        orientation = q_shortest_rotation([1,0,0], target)
    
    elif orient_type =='Fick' or orient_type =='nautical':
        n = np.atleast_2d(normalize(target))
        
        theta = np.arctan2(n[:,1], n[:,0])
        phi = -np.arcsin(n[:,2])
        
        orientation =  np.column_stack((theta, phi, np.zeros_like(theta)))
        orientation = np.rad2deg(orientation)
        
    elif orient_type == 'Helmholtz':
        n = np.atleast_2d(normalize(target))
        
        phi = -np.arctan2(n[:,2], n[:,0])
        theta = np.arcsin(n[:,1])
        
        orientation =  np.column_stack((phi, theta, np.zeros_like(theta)))
        orientation = np.rad2deg(orientation)
        
    else:
        raise ValueError('Input parameter {0} not known'.format(orientation))
    
    # For vector input, return a vector:
    if orientation.shape[0] == 1:
        orientation = orientation.ravel()
    
    return orientation
    
        
if __name__=='__main__':
    a = [3,3,0]
    b = [0, 1, 0]
    
    projected = project(a,b, projection_type='2D')
    print(projected)
   
