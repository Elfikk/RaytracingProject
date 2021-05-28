from numpy import inf

#Need definining: Reflectivity, Transmitivity
#Probably could use inheritance here actually, from master Object.

class Sphere():

    def __init__(self, position, radius, colour, reflectivity = 1, \
         transmitivity = 0):
        #__position is the centre of the sphere. Position should be a
        #numpy 3x1 array, radius any floating point number (no complex 
        #radii cmon).
        self.__position = position
        self.__radius = radius
        self.__colour = colour
        self.__reflectivity = reflectivity
        self.__transmitivity = transmitivity

    def get_radius(self):
        return self.__radius

    def get_position(self):
        return self.__position

    def get_position(self):
        return self.__colour

    def get_reflectivity(self):
        return self.__reflectivity

    def get_transmitivity(self):
        return self.__transmitivity

    def get_normal(self, position):
        relative_position =  self.get_position() - position
        return relative_position / self.get_radius()


class Plane():

    def __init__(self, normal, d, colour, limits = [-inf, inf, -inf,\
        inf, -inf, inf]):
        #r.n = d form. 
        #Normal is 3x1 array, d a float. Limits formatting:
        #[x_min, x_max, y_min, y_max, z_min, z_max].
        self.__normal = normal
        self.__constant = d
        self.__limits = limits
        self.__colour = colour

    def get_normal(self):
        return self.__normal
    
    def get_d(self):
        return self.__constant

    def get_limits(self):
        return self.__limits

    def get_colour(self):
        return self.__colour