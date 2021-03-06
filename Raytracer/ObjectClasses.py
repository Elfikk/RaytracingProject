import numpy as np
#import RayClass as rc
from RefractionMethods import sellmeier_n
   
class SceneObject():
    #All scene objects inherit from this parent. 

    def __init__(self, colour, reflectivity = 1, transmitivity = 0, \
        refractive_index = 1, sellmeier_Bs = np.array([]), sellmeier_Cs =\
             np.array([])):
        #Reflectivity is a multiplier for reflected ray intensities, and 
        #transmitivity for refracted rays. Can't create energy, 
        #so reflectivity + transmitivity <= 1.
        self.__colour = colour
        self.__reflectivity = reflectivity 
        self.__transmitivity = transmitivity
        self.__refractive_index = refractive_index
        self.__Bs = sellmeier_Bs
        self.__Cs = sellmeier_Cs

    def get_colour(self):
        return self.__colour

    def get_reflectivity(self):
        return self.__reflectivity

    def get_transmitivity(self):
        return self.__transmitivity

    def get_refractive_index(self):
        return self.__refractive_index

    def functional_n(self, wavelength):
        return sellmeier_n(wavelength, self.__Bs, self.__Cs)

    def intersect(self, ray):
        #This should never need to be called, but prevents crashes 
        #if you forget to define your intersection function.
        return np.inf

class Sphere(SceneObject):

    def __init__(self, position, radius, colour, reflectivity = 1, \
         transmitivity = 0, refractive_index = 1, sellmeier_Bs = np.array([]), \
             sellmeier_Cs = np.array([])):
        #__position is the centre of the sphere. Position should be a
        #numpy 3x1 array, radius any positive floating point number 
        #(no complex radii cmon).
        SceneObject.__init__(self, colour, reflectivity, transmitivity,\
            refractive_index, sellmeier_Bs, sellmeier_Cs)
        self.__position = position
        self.__radius = radius
        self.__type = 'sphere'

    def get_radius(self):
        return self.__radius

    def get_position(self):
        return self.__position

    def get_type(self):
        return self.__type

    #Needed for raindrop.
    def set_centre(self, new_pos):
        self.__position = new_pos
        
    def get_normal(self, position):
        #Returns the normalised vector between the centre of the sphere
        #and the given position (outward normal). Note that this in fact
        #does not require the position to be on the sphere!
        normal = position - self.get_position()
        return normal / np.linalg.norm(normal) #A lot of normals

    #calculating t for intersection of ray with sphere
    def intersect(self, ray):
        position = np.array(self.get_position())
        radius = self.get_radius()
        ray_position = np.array(ray.get_position_vector())
        ray_dir = np.array(ray.get_direction_vector())
        b = 2 * np.dot(ray_dir, ray_position - position)
        c = np.linalg.norm(ray_position - position) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c #Well spotted with the no need for a.
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            #t1 > t2 always, (neg + pos / pos) > (neg + neg / pos)
            if t2 > 0: #if t2 > 0, then so is t1, so t2 is min.
                #print('t2')
                return t2
            elif t1 > 0: #if t2 < 0, then we want t1.
                #print('t1')
                return t1
        return np.inf

class Plane(SceneObject):

    def __init__(self, normal, plane_position, colour, reflectivity = 1, \
        transmitivity = 0, refractive_index = 1, limits = [-np.inf, np.inf, \
         -np.inf, np.inf, -np.inf, np.inf], sellmeier_Bs = np.array([]), \
             sellmeier_Cs = np.array([])):
        #r.n = r.a form, where a is some arbitrary position on the plane.
        #Normal and position are 3x1 arrays. Limits formatting:
        #[x_min, x_max, y_min, y_max, z_min, z_max].
        SceneObject.__init__(self, colour, reflectivity, transmitivity, \
            refractive_index, sellmeier_Bs, sellmeier_Cs)

        #Numpy is nice enough to automatically change types when calling 
        #in-built functions like this - was causing me trouble last time,
        #and yet again when I refactored the classes today.
        self.__normal = normal/np.linalg.norm(normal) 

        self.__plane_position = plane_position
        self.__limits = limits
        self.__type = 'plane'

    # def get_normal(self, position = np.array([0, 0, -1000])):
    #     #The default is very unlikely to lay on a plane. It's possible,
    #     # but we should be fine.
    #     orientation_vector = self.get_plane_position() - position
    #     orientation_vector = orientation_vector/np.linalg.norm(\
    #         orientation_vector)
    #     # print(self.__normal, orientation_vector)
    #     # print(np.sign(np.dot(orientation_vector, self.__normal)))
    #     #(np.dot(orientation_vector, self.__normal), np.sign(np.dot(orientation_vector, self.__normal)))
    #     return int(np.sign(np.dot(orientation_vector, self.__normal)))\
    #          * self.__normal

    def get_normal(self, position = None):
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

class Circle(Plane):

    def __init__(self, normal, centre, colour, reflectivity = 1, \
        transmitivity = 0, refractive_index = 1, radius = 1, sellmeier_Bs\
        = np.array([]), sellmeier_Cs = np.array([])):
        
        Plane.__init__(self, normal, centre, colour, reflectivity, \
        transmitivity, refractive_index, sellmeier_Bs= sellmeier_Bs, \
        sellmeier_Cs = sellmeier_Cs)
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def intersect(self, ray):
        t = Plane.intersect(self, ray)
        if t != np.inf:
            intersection = ray.get_position(t)
            distance = np.sqrt(np.sum((self.get_plane_position() - \
                intersection)**2))
            if distance > self.get_radius():
                return np.inf
            return t
        return np.inf

if __name__ == '__main__':
    plane = Plane([100,0,5], [150,200,0], [0.7,0.2,0.2], 0.2)
    print(plane.get_reflectivity())

    sphere = Sphere(np.array([1,1,1]), 3, (0.1,0.1,0))
    print(sphere.get_normal(np.array([0,1,0])))

    #Rays can reach the god damn second thingy.
    # ray = rc.Ray(np.array([0,0,0]), np.array([1,0,0]))
    # ray_sphere = Sphere(np.array([3,0,0]), 1, (0,0,0))
    # print(ray_sphere.intersect(ray))
    # So why do they not in the renderer?!