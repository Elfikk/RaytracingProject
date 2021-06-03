import numpy as np

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
    
    def reflected_ray(self, object, distance):
        dir = self.get_direction_vector()
        reflected_pos = self.get_position_vector() + distance * self.get_direction_vector()
        if object.get_type() == 'plane':
            normal = object.get_normal()
        if object.get_type() == 'sphere':
            normal = object.get_normal(reflected_pos)
        reflected_dir = dir - 2 * normal * np.dot(dir, normal)
        reflected_ray = Ray(reflected_pos, reflected_dir)
        return reflected_ray