import pygame
import numpy as np
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import pandas as pd
import abc

# To ensure that the relative path works
import os
import sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) ) 
#dir_name = os.path.dirname(__file__)
#sys.path.append(os.path.realpath(os.path.join(dir_name, "..")))

from skinematics.sensors.xsens import XSens

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
    (4,5),
    )

colors = (
    (0.8,0,0),
    (0.7,0.7,0.6),
    (1,1,1),
    )

surfaces = (
    (0,1,2),
    (3,4,5),
    (0,1,3,4),
    (1,4,2,5),
    (0,3,2,5)
    )

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
    (4,5)
    )

def draw_axes():
    glBegin(GL_LINES)
    #glLineWidth(1.5)
    glColor3fv(colors[2])
    for line in axes:
        for vertex in line:
            glVertex3fv(axes_endpts[vertex])
    glEnd()

def Cube():
    glBegin(GL_TRIANGLES)

    for vertex in surfaces[1]:
        glColor3fv(colors[1])
        glVertex3fv(verticies[vertex])

    for vertex in surfaces[0]:
        glColor3fv(colors[0])
        glVertex3fv(verticies[vertex])
    #x = 0
    #for surface in surfaces:
        #for vertex in surface:
            #x+=1
            #glColor3fv(colors[np.mod(x,11)])
            #glVertex3fv(verticies[vertex])

    glEnd()

    glBegin(GL_LINES)
    glColor3fv(colors[2])
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])

    glEnd()
'''
'''


def main():
    # Get the data
    my_sensor = XSens(in_file=r'.\tests\data\data_xsens.txt')
    
    cameraX, cameraY, cameraZ = 0.2,0.2,0

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # Geänderter Code
    glMatrixMode(GL_PROJECTION) #Zuständig für Eigenschaften der Kamera z.B. Brennweite der Linse
    glLoadIdentity()
    # Ab hier wieder unverändert
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glLoadIdentity() # Ich würde diese Zeile löschen, weil sie vermutlich die ab Zeile 124 eingestellte Projektionsmatrix zurücksetzen würde.
        angle += 1
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # Geänderter Code
        glMatrixMode(GL_MODELVIEW) # Unter anderem zuständig fuer Position der Kamera
        glLoadIdentity()
        gluLookAt(
        cameraX, cameraY, cameraZ, # X, Y und Z Koordinaten der Kamera
             0, -0.2, 0.2, # Punkt(XYZ) auf den die Kamera schaut.
             0, 1, 0) # Definiert wo bei dieser Kamera "oben" ist.

        # Ab hier wieder unverändert
        glPushMatrix()
        glRotatef(angle, 1, 2, 1) # Der Uebergebene Vektor sollte ein normalisierter Vektor sein. Wenn nicht wird er von OpenGL normalisiert.
        Cube()
        glPopMatrix()
        draw_axes()
        pygame.display.flip()
        pygame.time.wait(10)

main()
