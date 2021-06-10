import numpy as np

#Need definining: Reflectivity, Transmitivity
   
class SceneObject():
    #All scene objects inherit from this parent. 

    def __init__(self, colour, reflectivity = 1, transmitivity = 0, \
        refractive_index = 1):
        self.__colour = colour
        self.__reflectivity = reflectivity
        self.__transmitivity = transmitivity
        self.__refractive_index = refractive_index

    def get_colour(self):
        return self.__colour

    def get_reflectivity(self):
        return self.__reflectivity

    def get_transmitivity(self):
        return self.__transmitivity

    def get_refractive_index(self):
        return self.__refractive_index

    def intersect(self, ray):
        #This should never need to be called, but prevents crashes 
        #if you forget to define your intersection function.
        return np.inf

class Sphere(SceneObject):

    def __init__(self, position, radius, colour, reflectivity = 1, \
         transmitivity = 0, refractive_index = 1):
        #__position is the centre of the sphere. Position should be a
        #numpy 3x1 array, radius any positive floating point number 
        #(no complex radii cmon).
        SceneObject.__init__(self, colour, reflectivity, transmitivity,\
            refractive_index)
        self.__position = position
        self.__radius = radius
        self.__type = 'sphere'

    def get_radius(self):
        return self.__radius

    def get_position(self):
        return self.__position

    def get_type(self):
        return self.__type
        
    def get_normal(self, position):
        #Returns the normalised vector between the centre of the sphere
        #and the given position (outward normal). Note that this in fact
        #does not require the position to be on the sphere!
        normal = position - self.get_position()
        return normal / np.linalg.norm(normal) #A lot of normals

    #calculating t for intersection of ray with sphere
    def intersect(self, ray):
        # a is equal to one
        position = np.array(self.get_position())
        radius = self.get_radius()
        ray_position = np.array(ray.get_position_vector())
        ray_dir = np.array(ray.get_direction_vector())
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

class Plane(SceneObject):

    def __init__(self, normal, plane_position, colour, reflectivity = 1, \
        transmitivity = 0, refractive_index = 1, limits = [-np.inf, np.inf, \
         -np.inf, np.inf, -np.inf, np.inf]):
        #r.n = r.a form, where a is some arbitrary position on the plane.
        #Normal and position are 3x1 arrays. Limits formatting:
        #[x_min, x_max, y_min, y_max, z_min, z_max].
        SceneObject.__init__(self, colour, reflectivity, transmitivity, \
            refractive_index)

        #Numpy is nice enough to automatically change types when calling 
        #in-built functions like this - was causing me trouble last time,
        #and yet again when I refactored the classes today.
        self.__normal = normal/np.linalg.norm(normal) 

        self.__plane_position = plane_position
        self.__limits = limits
        self.__type = 'plane'

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

if __name__ == '__main__':
    plane = Plane([100,0,5], [150,200,0], [0.7,0.2,0.2], 0.2)
    print(plane.get_reflectivity())

    sphere = Sphere(np.array([1,1,1]), 3, (0.1,0.1,0))
    print(sphere.get_normal(np.array([0,1,0])))