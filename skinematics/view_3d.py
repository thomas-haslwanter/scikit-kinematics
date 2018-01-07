'''
Alternative 3D Viewer, based on OpenGL.
Badly hacked, since I don't really know OpenGL. Improvements would be VERY welcome!
'''

'''
author: Thomas Haslwanter
date:   Jan 2018
'''

import pygame
import numpy as np

import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu

import numpy as np
import pandas as pd

# To ensure that the relative path works
import os
import sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) ) 

from skinematics.sensors.xsens import XSens
from skinematics.vector import rotate_vector

def define_elements():
    # Define the elements
    delta = 0.01
    vertices = (
        (0, -0.2, delta),
        (0, 0.2, delta),
        (0.6, 0, delta),
        (0, -0.2, -delta),
        (0, 0.2, -delta),
        (0.6, 0, -delta),
        )
    
    edges = (
        (0,1),
        (0,2),
        (0,3),
        (1,2),
        (1,4),
        (2,5),
        (3,4),
        (3,5),
        (4,5) )
    
    colors = (
        (0.8,0,0),
        (0.7,0.7,0.6),
        (1,1,1) )
    
    surfaces = (
        (0,1,2),
        (3,4,5),
        (0,1,3,4),
        (1,4,2,5),
        (0,3,2,5) )
    
    axes_endpts = np.array(
        [[-1,  0,  0],
         [ 1,  0,  0],
         [ 0, -1,  0],
         [ 0,  1,  0],
         [ 0,  0, -1],
         [ 0,  0,  1]])
    
    axes = (
        (0,1),
        (2,3),
        (4,5) )
    
    return (vertices, surfaces, edges, colors, axes, axes_endpts)

def draw_axes(axes):
    '''Draw the axes.
    Here I have a difficulty with making the axes thicker'''
    
    gl.glBegin(gl.GL_LINES)
    gl.glColor3fv(colors[2])
    #glLineWidth(1.5)
    
    for line in axes:
        for vertex in line:
            gl.glVertex3fv(axes_endpts[vertex])
            
    gl.glEnd()

def draw_pointer(colors, surfaces, edges, vertices):
    '''Draw the triangle that indicates 3D orientation'''
    
    gl.glBegin(gl.GL_TRIANGLES)

    for (color, surface) in zip(colors[:2], surfaces[:2]):
        for vertex in surface:
            gl.glColor3fv(color)
            gl.glVertex3fv(vertices[vertex])
    gl.glEnd()

    gl.glBegin(gl.GL_LINES)
    gl.glColor3fv(colors[2])
    
    for edge in edges:
        for vertex in edge:
            gl.glVertex3fv(vertices[vertex])
    gl.glEnd()
    

if __name__ == '__main__':
    # Get the data
    my_sensor = XSens(in_file=r'.\tests\data\data_xsens.txt')
    
    # Define the plotting elements
    vertices, surfaces, edges, colors, axes, axes_endpts = define_elements()
    
    # Define the view-point (=camera_position) and the gaze_target
    camera = [0.2, 0.2, 0]
    gaze_target = [0, 0, -1]
    camera_up = [0, 1, 0]
    
    # OpenGL to my convention
    x = [1, 0, 0]
    y = [0, 0, 1]
    z = [0, 1, 0]
    openGL2skin = np.column_stack( (x,y,z) )
    
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF|pygame.OPENGL)

    # Camera properties, e.g. focal length etc
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    
    glu.gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gl.glTranslatef(0.0,0.0, -3)

    loop_index = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

                
        # Re-set the projection-matrix from line 130??
        #glLoadIdentity() 
        
        quat = my_sensor.quat
        #quat = np.zeros( (1000, 3) )
        #from positive_rotations import make_positive_rotations
        #quat = make_positive_rotations()
        loop_index = np.mod(loop_index+1, quat.shape[0])
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Camera position
        gl.glMatrixMode(gl.GL_MODELVIEW) 
        gl.glLoadIdentity()
        glu.gluLookAt(
            camera[0], camera[1], camera[2],
            gaze_target[0], gaze_target[1], gaze_target[2], 
            camera_up[0], camera_up[1], camera_up[2] )

        # Scene elements
        gl.glPushMatrix()
        cur_corners = rotate_vector(vertices, quat[loop_index]) @ openGL2skin.T
        cur_corners = cur_corners * np.r_[1, 1, -1] # This seems to be required
                    #to get things right - but I don't understand OpenGL at this point
        
        draw_pointer(colors, surfaces, edges, cur_corners)
        gl.glPopMatrix()
        draw_axes(axes)
        
        pygame.display.flip()
        pygame.time.wait(10)