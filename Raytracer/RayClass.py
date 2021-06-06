import numpy as np
import RefractionMethods as rm

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

    def get_position(self, t):
        #Line equation: r = p + dt. Given a value of t, returns a
        #position on the ray's trajectory.
        return self.get_position_vector() + t * self.get_direction_vector()
    
    #I don't like having the secondary ray firing from here, but it'll stay
    #that way for now, until the Scene takes over the handling. Same case 
    #with refraction of rays - better to stay consistently shit then being
    #completely inconsistent. Easier to wrap your head around it :)
    def reflected_ray(self, object, distance):
        dir = self.get_direction_vector()
        reflected_pos = self.get_position_vector() + distance * \
            self.get_direction_vector()
        normal = object.get_normal(reflected_pos)
        reflected_dir = dir - 2.0 * normal * np.dot(dir, normal)
        reflected_ray = Ray(reflected_pos, reflected_dir)
        return reflected_ray

    def refracted_ray(self, object, intersection, n1 = 1):
        normal = object.get_normal(intersection)
        direction = rm.refracted_direction(self.get_direction_vector(), \
            normal, object.get_refractive_index(), n1)
        return Ray(intersection, direction)