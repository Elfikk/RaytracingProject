import numpy as np

#Need definining: Reflectivity, Transmitivity
#Probably could use inheritance here actually, from master Object.
   
class SceneObject():

    def __init__(self, colour, reflectivity = 1, transmitivity = 0):
        self.__colour = colour
        self.__reflectivity = reflectivity
        self.__transmitivity = transmitivity

    def get_colour(self):
        return self.__colour

    def get_reflectivity(self):
        return self.__reflectivity

    def get_transmitivity(self):
        return self.__transmitivity

    def intersect(self, ray):
        print('a')
        return np.inf

class Sphere(SceneObject):

    def __init__(self, position, radius, colour, reflectivity = 1, \
         transmitivity = 0):
        #__position is the centre of the sphere. Position should be a
        #numpy 3x1 array, radius any floating point number (no complex
        #radii cmon).
        SceneObject.__init__(self, colour, reflectivity, transmitivity)
        self.__position = position
        self.__radius = radius
        self.__type = 'sphere'

    def get_radius(self):
        return self.__radius

    def get_position(self):
        return self.__position

    def get_type(self):
        return self.__type

    def intersect(self,ray):
        # a is equal to one
        position = np.array(self.get_position())
        radius = self.get_radius()
        ray_position = np.array(ray.get_position_vector())
        ray_dir = np.array(ray.get_direction_vector())
        a = np.dot(ray_dir,ray_dir)
        b = 2 * np.dot(ray_dir, ray_position - position)
        c = np.linalg.norm(ray_position - position) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c* a
        print(delta)
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2*a)
            t2 = (-b - np.sqrt(delta)) / (2*a)
            print(t1, t2)
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return np.inf

class Plane(SceneObject):

    def __init__(self, normal, plane_position, colour, \
        limits = [-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf], \
        reflectivity = 1, transmitivity = 0):
        #r.n = d form.
        #Normal is 3x1 array, d a float. Limits formatting:
        #[x_min, x_max, y_min, y_max, z_min, z_max].
        SceneObject.__init__(self, colour, reflectivity, transmitivity)
        self.__normal = normal
        self.__plane_position = plane_position
        self.__limits = limits
        self.__type = 'plane'

    def get_normal(self):
        return self.__normal

    def get_plane_position(self):
        return self.__plane_position

    def get_limits(self):
        return self.__limits

    def get_type(self):
        return self.__type

    def intersect(self, ray):
        ray_position = np.array(ray.get_position_vector())
        ray_dir = np.array(ray.get_direction_vector())
        plane_position = np.array(self.get_plane_position())
        plane_norm = np.array(self.get_normal())
        delta = np.dot(ray_dir,plane_norm)
        if np.abs(delta) < 1e-6:
            return np.inf
        t = np.dot(plane_position-ray_position,plane_norm)/delta
        if t < 0:
            return np.inf
        return t