#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:52:55 2021

@author: ioanabalabasciuc
"""
import numpy as np
import matplotlib.pyplot as plt
from RayClass import Ray
import SceneClass as Scene
from ObjectClasses import Sphere, Plane

objects = [Sphere([0,0,500],500,[0.1,0.5,0.2])] # [Plane([100,0,5],[150,200,0],[0.2,0.1,0.22]), 
 
def nearest_intersect_objects(ray, objects):
    distances = []
    for object in objects:
        distances.append(object.intersect(ray)) 
    if np.min(distances) == np.inf:
        return [0,0,0]
    else:
        i = distances.index(np.min(distances))
        return objects[i].get_colour()

image = np.zeros([300,400,3])
for i in range(300):
    print(i)
    for j in range(400):
        ray = Ray([150,200,-500],[i-150,j-200,100])
        colour = nearest_intersect_objects(ray,objects)
        image[i,j] = colour
plt.imshow(image)
plt.show()
    