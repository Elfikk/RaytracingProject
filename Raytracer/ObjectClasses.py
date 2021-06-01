import numpy as np
from numpy import inf

#Need definining: Reflectivity, Transmitivity
#Probably could use inheritance here actually, from master Object.
   
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