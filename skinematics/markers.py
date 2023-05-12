'''
Utilities for analyzing movement data recorded with marker-based video
systems.
'''

'''
Author: Thomas Haslwanter
Version: 1.2
Date: Aug-2017
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd 
from numpy import r_, sum

# The following construct is required since I want to run the module as a script
# inside the skinematics-directory
import os
import sys

file_dir = os.path.dirname(__file__)
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

import quat, vector

def analyze_3Dmarkers(MarkerPos, ReferencePos):
    '''
    Take recorded positions from 3 markers, and calculate center-of-mass (COM) and orientation
    Can be used e.g. for the analysis of Optotrac data.

    Parameters
    ----------
    MarkerPos : ndarray, shape (N,9)
        x/y/z coordinates of 3 markers

    ReferencePos : ndarray, shape (1,9)
        x/y/z coordinates of markers in the reference position

    Returns
    -------
    Position : ndarray, shape (N,3)
        x/y/z coordinates of COM, relative to the reference position
    Orientation : ndarray, shape (N,3)
        Orientation relative to reference orientation, expressed as quaternion

    Example
    -------
    >>> (PosOut, OrientOut) = analyze_3Dmarkers(MarkerPos, ReferencePos)


    '''

    # Specify where the x-, y-, and z-position are located in the data
    cols = {'x' : r_[(0,3,6)]} 
    cols['y'] = cols['x'] + 1
    cols['z'] = cols['x'] + 2    

    # Calculate the position
    cog = np.vstack(( sum(MarkerPos[:,cols['x']], axis=1),
                      sum(MarkerPos[:,cols['y']], axis=1),
                      sum(MarkerPos[:,cols['z']], axis=1) )).T/3.

    cog_ref = np.vstack(( sum(ReferencePos[cols['x']]),
                          sum(ReferencePos[cols['y']]),
                          sum(ReferencePos[cols['z']]) )).T/3.                      

    position = cog - cog_ref    

    # Calculate the orientation    
    numPoints = len(MarkerPos)
    orientation = np.ones((numPoints,3))

    refOrientation = vector.plane_orientation(ReferencePos[:3], ReferencePos[3:6], ReferencePos[6:])

    for ii in range(numPoints):
        '''The three points define a triangle. The first rotation is such
        that the orientation of the reference-triangle, defined as the
        direction perpendicular to the triangle, is rotated along the shortest
        path to the current orientation.
        In other words, this is a rotation outside the plane spanned by the three
        marker points.'''

        curOrientation = vector.plane_orientation( MarkerPos[ii,:3], MarkerPos[ii,3:6], MarkerPos[ii,6:])
        alpha = vector.angle(refOrientation, curOrientation)        

        if alpha > 0:
            n1 = vector.normalize( np.cross(refOrientation, curOrientation) )
            q1 = n1 * np.sin(alpha/2)
        else:
            q1 = r_[0,0,0]

        # Now rotate the triangle into this orientation ...
        refPos_after_q1 = vector.rotate_vector(np.reshape(ReferencePos,(3,3)), q1)

        '''Find which further rotation in the plane spanned by the three marker points
	is necessary to bring the data into the measured orientation.'''

        Marker_0 = MarkerPos[ii,:3]
        Marker_1 = MarkerPos[ii,3:6]
        Vector10 = Marker_0 - Marker_1
        vector10_ref = refPos_after_q1[0]-refPos_after_q1[1]
        beta = vector.angle(Vector10, vector10_ref)

        q2 = curOrientation * np.sin(beta/2)

        if np.cross(vector10_ref,Vector10).dot(curOrientation)<=0:
            q2 = -q2
        orientation[ii,:] = quat.q_mult(q2, q1)

    return (position, orientation)

def find_trajectory(r0, Position, Orientation):
    '''
    Movement trajetory of a point on an object, from the position and
    orientation of a sensor, and the relative position of the point at t=0.

    Parameters
    ----------
    r0 : ndarray (3,)
        Position of point relative to center of markers, when the object is
        in the reference position.
    Position : ndarray, shape (N,3)
        x/y/z coordinates of COM, relative to the reference position
    Orientation : ndarray, shape (N,3)
        Orientation relative to reference orientation, expressed as quaternion

    Returns
    -------
    mov : ndarray, shape (N,3)
        x/y/z coordinates of the position on the object, relative to the
        reference position of the markers

    Notes
    ----- 

      .. math::

          \\vec C(t) = \\vec M(t) + \\vec r(t) = \\vec M(t) +
          {{\\bf{R}}_{mov}}(t) \\cdot \\vec r({t_0})

    Examples
    --------
    >>> t = np.arange(0,10,0.1)
    >>> translation = (np.c_[[1,1,0]]*t).T
    >>> M = np.empty((3,3))
    >>> M[0] = np.r_[0,0,0]
    >>> M[1]= np.r_[1,0,0]
    >>> M[2] = np.r_[1,1,0]
    >>> M -= np.mean(M, axis=0) 
    >>> q = np.vstack((np.zeros_like(t), np.zeros_like(t),quat.deg2quat(100*t))).T
    >>> M0 = vector.rotate_vector(M[0], q) + translation
    >>> M1 = vector.rotate_vector(M[1], q) + translation
    >>> M2 = vector.rotate_vector(M[2], q) + translation
    >>> data = np.hstack((M0,M1,M2))
    >>> (pos, ori) = signals.analyze_3Dmarkers(data, data[0])
    >>> r0 = np.r_[1,2,3]
    >>> movement = find_trajectory(r0, pos, ori)

    '''

    return Position + vector.rotate_vector(r0, Orientation)

if __name__ == '__main__':
    # Here a marker-test is required

    print('Done')
