#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:52:55 2021

@author: ioanabalabasciuc
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from RayClass import Ray
from SceneClass import Scene
from ObjectClasses import Sphere, Plane

objects = [Plane([100,0,5], [150,200,0], [0.7,0.2,0.2], 0.2), \
           Sphere([-200,-20,200], 300, [0.2,0.7,0.2], 0.7), \
           Sphere([-100,500,200], 250, [0.2,0.2,0.7], 0.5)]
 
def nearest_intersect_objects(ray, objects):
    distances = []
    for object in objects:
        if object.get_type() == 'sphere':
            distances.append(object.sphere_intersect(ray)) 
        elif object.get_type() == 'plane':
            distances.append(object.plane_intersect(ray))
    if np.min(distances) == np.inf:
        return np.inf
    else:
        i = distances.index(np.min(distances))
        return [objects[i], distances[i]]

def colour(ray, objects):
    if nearest_intersect_objects(ray,objects) == np.inf:
        return [0,0,0]
    else:
        object = nearest_intersect_objects(ray,objects)[0]
        distance = nearest_intersect_objects(ray,objects)[1]
        reflectivity = object.get_reflectivity()
        colour = (1-reflectivity) * np.array(object.get_colour())
        for i in range(5):
            reflected_ray = ray.reflected_ray(object, distance)
            if nearest_intersect_objects(reflected_ray, objects) == np.inf:
                break
            else:
                object = nearest_intersect_objects(reflected_ray,objects)[0]
                distance = nearest_intersect_objects(reflected_ray,objects)[1]
                reflected_colour = object.get_colour()
                colour = colour + reflectivity * np.array(reflected_colour)
                ray = reflected_ray
                reflectivity *= object.get_reflectivity()
            if reflectivity == 0:
                break
        return colour

image = np.zeros([300,400,3])
for i in tqdm(range(300)):
    for j in range(400):
        ray = Ray([150,200,-500],[i-150,j-200,100])
        image[i,j] = colour(ray, objects)
        tqdm._instances.clear()
plt.imshow(image)