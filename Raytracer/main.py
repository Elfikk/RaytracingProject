import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from RayClass import Ray
from SceneClass import Scene
from ObjectClasses import Sphere, Plane

# objects = [Plane([100,0,5], [150,200,0], [0.7,0.2,0.2], 0.2), \
#            Sphere([-200,-20,200], 300, [0.2,0.7,0.2], 0.7), \
#            Sphere([-100,500,200], 250, [0.2,0.2,0.7], 0.5)]

objects = [Plane([100,0,5], [150,200,0], [0.7,0.2,0.2], 0.2), \
           Sphere([0, 0, 250], 100, [0.2,0.7,0.2], 0.7)]#, \
        #    Plane([-1,0,1], [0,0,50], [0.,0.5,0.], transmitivity = 0.99,\
        #    refractive_index = 2)]
 
def nearest_intersect_objects(ray, objects):
    distances = []
    for object in objects:
        distances.append(object.intersect(ray))
    if np.min(distances) == np.inf:
        return np.inf
    else:
        i = distances.index(np.min(distances))
        return [objects[i], distances[i]]

def colour(ray, objects, max_depth = 5):
    if nearest_intersect_objects(ray,objects) == np.inf:
        return [0.53, 0.80 , 0.98]
    else:
        object = nearest_intersect_objects(ray,objects)[0]
        distance = nearest_intersect_objects(ray,objects)[1]
        reflectivity = object.get_reflectivity()
        colour = (1-reflectivity) * np.array(object.get_colour())
        for i in range(max_depth):
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

def refractive_rendering(ray, objects, max_depth = 3):
    #Currently separate to reflection - easier to write one thing at a
    #time. 
    if nearest_intersect_objects(ray,objects) == np.inf:
        return [0.53, 0.80 , 0.98] #I'm getting tired of the void
    else:
        object = nearest_intersect_objects(ray,objects)[0]
        distance = nearest_intersect_objects(ray,objects)[1]
        transmitivity = object.get_transmitivity()
        colour = (1-transmitivity) * np.array(object.get_colour())
        if transmitivity: #Only false for trans = 0
            for i in range(max_depth):
                refracted_ray = ray.refracted_ray(object, \
                    ray.get_position(distance))
                if nearest_intersect_objects(refracted_ray, objects) == np.inf:
                    colour += transmitivity * np.array([0.53, 0.80 , 0.98])
                    break
                else:
                    object = nearest_intersect_objects(refracted_ray,objects)[0]
                    distance = nearest_intersect_objects(refracted_ray,objects)[1]
                    refracted_colour = object.get_colour()
                    colour += transmitivity * np.array(refracted_colour)
                    transmitivity *= object.get_transmitivity()
                if transmitivity == 0:
                    break
                ray = refracted_ray
        return colour


image = np.zeros([300,400,3])
for i in tqdm(range(300)):
    for j in range(400):
        direction_vector = np.array([i-150, j-200, 0]) - np.array([150,200,-500])
        ray = Ray([150,200,-500], direction_vector)
        image[i,j] = colour(ray, objects)
        tqdm._instances.clear()
plt.imshow(image)
plt.show()