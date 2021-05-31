#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:52:55 2021

@author: ioanabalabasciuc
"""
import numpy as np
import matplotlib.pyplot as plt
class Ray():
#Class defining our funky rays.

    def __init__(self, pos_vector, dir_vector, wavelength = None, \
         intensity = 1):
        self.__position_vector = pos_vector
        self.__direction_vector = dir_vector/np.linalg.norm(dir_vector)
        self.__wavelength = wavelength
        self.__intensity = intensity

    def get_wavelength(self):
        return self.__wavelength

    def get_position_vector(self):
        return self.__position_vector

    def get_direction_vector(self):
        return self.__direction_vector

    def get_intensity(self):
        return self.__intensity
    
class Sphere():

    def __init__(self, position, radius, colour,type = 'sphere',reflectivity = 1, \
         transmitivity = 0):
        #__position is the centre of the sphere. Position should be a
        #numpy 3x1 array, radius any floating point number (no complex
        #radii cmon).
        self.__position = position
        self.__radius = radius
        self.__colour = colour
        self.__reflectivity = reflectivity
        self.__transmitivity = transmitivity
        self.__type = type

    def get_radius(self):
        return self.__radius

    def get_position(self):
        return self.__position

    def get_colour(self):
        return self.__colour

    def get_reflectivity(self):
        return self.__reflectivity

    def get_transmitivity(self):
        return self.__transmitivity

    def get_type(self):
        return self.__type

    #calculating t for intersection of ray with sphere
    def sphere_intersect(self,Ray):
        # a is equal to one
        position = np.array(self.get_position())
        radius = self.get_radius()
        ray_position = np.array(Ray.get_position_vector())
        ray_dir = np.array(Ray.get_direction_vector())
        a = np.dot(ray_dir,ray_dir)
        b = 2 * np.dot(ray_dir, ray_position - position)
        c = np.linalg.norm(ray_position - position) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c* a
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2*a)
            t2 = (-b - np.sqrt(delta)) / (2*a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return np.inf

class Plane():

    def __init__(self, normal, plane_position, colour, type = 'plane',limits = [-np.inf, np.inf, -np.inf,\
        np.inf, -np.inf, np.inf]):
        #r.n = d form.
        #Normal is 3x1 array, d a float. Limits formatting:
        #[x_min, x_max, y_min, y_max, z_min, z_max].
        self.__normal = normal
        self.__plane_position = plane_position
        self.__limits = limits
        self.__colour = colour
        self.__type = type 

    def get_normal(self):
        return self.__normal

    def get_plane_position(self):
        return self.__plane_position

    def get_limits(self):
        return self.__limits

    def get_colour(self):
        return self.__colour

    def get_type(self):
        return self.__type

    def plane_intersect(self,Ray):
        ray_position = np.array(Ray.get_position_vector())
        ray_dir = np.array(Ray.get_direction_vector())
        plane_position = np.array(self.get_plane_position())
        plane_norm = np.array(self.get_normal())
        delta = np.dot(ray_dir,plane_norm)
        if np.abs(delta) < 1e-6:
            return np.inf
        t = np.dot(plane_position-ray_position,plane_norm)/delta
        if t < 0:
            return np.inf
        return t

objects = [Plane([100,0,5],[150,200,0],[0.2,0.1,0.22]),Sphere([-200,-20,200],100,[0.1,0.5,0.2])]#Sphere([10,20,1],60,),]
 
def nearest_intersect_objects(ray, objects):
    distances = []
    for object in objects:
        if object.get_type() == 'sphere':
            distances.append(object.sphere_intersect(ray)) 
        elif object.get_type() == 'plane':
            distances.append(object.plane_intersect(ray))
    if np.min(distances) == np.inf:
        return [0,0,0]
    else:
        i = distances.index(np.min(distances))
        return objects[i].get_colour()

image = np.zeros([300,400,3])
for i in range(300):
    for j in range(400):
        ray = Ray([150,200,-500],[i-150,j-200,100])
        colour = nearest_intersect_objects(ray,objects)
        image[i,j] = colour
plt.imshow(image)
    