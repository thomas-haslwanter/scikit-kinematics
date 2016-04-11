'''
Routines for working with vectors
These routines can be used with vectors, as well as with matrices containing a vector in each row.
'''
 
'''
author :  Thomas Haslwanter
date :    Jan-2015
version : 1.5
'''

import numpy as np

# The following construct is required since I want to run the module as a script
# inside the thLib-directory
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import quat 

def normalize(v):
    ''' Normalization of a given vector 
    
    Parameters
    ----------
    v : array (N,) or (M,N)
        input vector

    Returns
    -------
    v_normalized : array (N,) or (M,N)
        normalized input vector

    Example
    -------
    >>> thLib.vector.normalize([3, 0, 0])
    array([[ 1.,  0.,  0.]])
    
    >>> v = [[pi, 2, 3], [2, 0, 0]]
    >>> thLib.vector.normalize(v)
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

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    >>>       [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    >>>       [0,1,0]])
    >>> thLib.vector.angle(v1,v2)
    array([ 1.30024656,  0.96453036])
    
    Notes
    -----

    .. math::
        \\alpha =arccos(\\frac{\\vec{v_1} \\cdot \\vec{v_2}}{| \\vec{v_1} |
        \\cdot | \\vec{v_2}|})

    '''
    
    if v1.ndim < v2.ndim:
        v1, v2 = v2, v1
    n1 = normalize(v1)
    n2 = normalize(v2)
    if v2.ndim == 1:
        angle = np.arccos(n1.dot(n2))
    else:
        angle = np.arccos(list(map(np.dot, n1, n2)))
    return angle
 
def project(v1,v2):
    '''Project one vector onto another
    
    Parameters
    ----------
    v1 : array (N,) or (M,N)
        projected vector
    v2 : array (N,) or (M,N)
        target vector

    Returns
    -------
    v_projected : array (N,) or (M,N)
        projection of v1 onto v2

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    >>>       [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    >>>       [0,1,0]])
    >>> thLib.vector.project(v1,v2)
    array([[ 1.,  0.,  0.],
       [ 0.,  5.,  0.]])
     
    Notes
    -----

    .. math::

        \\vec{n} = \\frac{ \\vec{a} }{| \\vec{a} |}

        \\vec{v}_{proj} = \\vec{n} (\\vec{v} \\cdot \\vec{n})

    '''
    
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)
    
    e2 = normalize(v2)
    if e2.ndim == 1 or e2.shape[0]==1:
        return (e2 * list(map(np.dot, v1, e2))).ravel()
    else:
        return (e2.T * list(map(np.dot, v1, e2))).T
 
def GramSchmidt(p1,p2,p3):
    '''Gram-Schmidt orthogonalization
    
    Parameters
    ----------
    p1 : array (3,) or (M,3)
        coordinates of Point 1
    p2 : array (3,) or (M,3)
        coordinates of Point 2
    p3 : array (3,) or (M,3)
        coordinates of Point 3

    Returns
    -------
    Rmat : array (9,) or (M,9)
        flattened rotation matrix

    Example
    -------
    >>> P1 = np.array([[0, 0, 0], [1,2,3]])
    >>> P2 = np.array([[1, 0, 0], [4,1,0]])
    >>> P3 = np.array([[1, 1, 0], [9,-1,1]])
    >>> GramSchmidt(P1,P2,P3)
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
    
    v1 = np.atleast_2d(p2-p1)
    v2 = np.atleast_2d(p3-p1)
        
    e1 = normalize(v1)
    e2 = normalize(v2- project(v2,e1))
    e3 = np.cross(e1,e2)
    
    return np.hstack((e1,e2,e3))

def plane_orientation(p1, p2, p3):
    '''The vector perpendicular to the plane defined by three points.
    
    Parameters
    ----------
    p1 : array (3,) or (M,3)
        coordinates of Point 1
    p2 : array (3,) or (M,3)
        coordinates of Point 2
    p3 : array (3,) or (M,3)
        coordinates of Point 3

    Returns
    -------
    n : array (3,) or (M,3)
        vector perpendicular to the plane

    Example
    -------
    >>> P1 = np.array([[0, 0, 0], [1,2,3]])
    >>> P2 = np.array([[1, 0, 0], [4,1,0]])
    >>> P3 = np.array([[1, 1, 0], [9,-1,1]])
    >>> plane_orientation(P1,P2,P3)
    array([[ 0.        ,  0.        ,  1.        ],
           [-0.36196138, -0.93075784, -0.05170877]])

    Notes
    -----

    .. math::
        \\vec{n} = \\frac{ \\vec{a} \\times \\vec{b}} {| \\vec{a} \\times \\vec{b}|}
    

    '''

    v12 = p2-p1
    v13 = p3-p2
    n = np.cross(v12,v13)
    return normalize(n)

def qrotate(v1,v2):
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
    vRotated = quat.quatmult(q, quat.quatmult(qvector, quat.quatinv(q)))
    vRotated = vRotated[:,1:]

    if min(vRotated.shape)==1:
        vRotated = vRotated.ravel()

    return vRotated

if __name__=='__main__':
    print(normalize([3, 0, 0]))
   