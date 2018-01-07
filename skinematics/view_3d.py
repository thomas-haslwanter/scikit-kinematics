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
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import pandas as pd
#import abc

# To ensure that the relative path works
import os
import sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) ) 

from skinematics.sensors.xsens import XSens

def define_elements():
    # Define the elements
    delta = 0.01
    verticies = (
        (0, -0.2, -delta),
        (0, 0.2, -delta),
        (0.6, 0, -delta),
        (0, -0.2, delta),
        (0, 0.2, delta),
        (0.6, 0, delta),
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
    
    return (verticies, surfaces, edges, colors, axes, axes_endpts)

def draw_axes(axes):
    '''Draw the axes.
    Here I have a difficulty with making the axes thicker'''
    
    glBegin(GL_LINES)
    glColor3fv(colors[2])
    #glLineWidth(1.5)
    
    for line in axes:
        for vertex in line:
            glVertex3fv(axes_endpts[vertex])
            
    glEnd()

def draw_pointer(colors, surfaces, edges):
    '''Draw the triangle that indicates 3D orientation'''
    
    glBegin(GL_TRIANGLES)

    for (color, surface) in zip(colors[:2], surfaces[:2]):
        for vertex in surface:
            glColor3fv(color)
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    glColor3fv(colors[2])
    
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


if __name__ == '__main__':
    # Get the data
    my_sensor = XSens(in_file=r'.\tests\data\data_xsens.txt')
    
    # Define the plotting elements
    verticies, surfaces, edges, colors, axes, axes_endpts = define_elements()
    
    # Define the view-point (=camera_position) and the gaze_target
    camera = [0.2, 0.2, 0]
    gaze_target = [0, -0.2, 0.2]
    camera_up = [0, 1, 0]

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # Camera properties, e.g. focal length etc
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

                
        # Re-set the projection-matrix from line 130??
        #glLoadIdentity() 
        
        angle += 1
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # Camera position
        glMatrixMode(GL_MODELVIEW) 
        glLoadIdentity()
        gluLookAt(
            camera[0], camera[1], camera[2],
            gaze_target[0], gaze_target[1], gaze_target[2], 
            camera_up[0], camera_up[1], camera_up[2] )

        # Scene elements
        glPushMatrix()
        glRotatef(angle, 1, 2, 1) # Der Uebergebene Vektor sollte ein normalisierter Vektor sein. Wenn nicht wird er von OpenGL normalisiert.
        draw_pointer(colors, surfaces, edges)
        glPopMatrix()
        draw_axes(axes)
        
        pygame.display.flip()
        pygame.time.wait(10)