import numpy as np
from RayClass import Ray

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
        b = 2 * np.dot(ray_dir, ray_position - position)
        c = np.linalg.norm(ray_position - position) ** 2 - radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
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

class lens(SceneObject):

    def __init__(self, position_lens, direction_lens, radius_lens, \
                 thickness, focal_length, colour, reflectivity = 0, \
         transmitivity = 1, refractive_index = 1.517):
        # Position is the centre of the cicle at the base of the parabaloid.
        # Position should be a numpy 3x1 array
        SceneObject.__init__(self, colour, reflectivity, transmitivity,\
            refractive_index)
        self.__radius_lens = radius_lens
        self.__thickness = thickness
        self.__focal_length = focal_length
        self.__position_lens = position_lens
        self.__direction_lens = direction_lens/np.linalg.norm(direction_lens)
        self.__type = 'lens'
    
    def get_radius_lens(self):
        return self.__radius_lens

    def get_thickness(self):
        return self.__thickness
    
    def get_focal_length(self):
        return self.__focal_length

    def get_position_lens(self):
        return self.__position_lens
    
    def get_direction_lens(self):
        return self.__direction_lens

    def get_type(self):
        return self.__type
        
    def get_normal_plane(self, position = None):
        return self.get_direction_lens()
    
    def get_normal_sphere(self, position):
        position_lens = np.array(self.get_position_lens())
        direction_lens = np.array(self.get_direction_lens())
        t_0 = self.get_thickness()
        radius_sphere = self.get_focal_length() * (1.517-1)
        position_sphere = position_lens + (t_0-radius_sphere) * direction_lens
        sphere = Sphere(position_sphere, radius_sphere, [0,0,0], 0, 1, 1.517)

    def intersect(self,ray):
        plane = self.get_plane()
        sphere = self.get_sphere()
        ray_position = np.array(ray.get_position_vector())
        ray_dir = np.array(ray.get_direction_vector())
        ray_center_1 = Ray(sphere.get_position(),plane.get_normal())
        ray_center_2 = Ray(sphere.get_position(),-1 * plane.get_normal())
        distance_1= plane.intersect(ray_center_1)
        distance_2= plane.intersect(ray_center_2)
        distance = min(distance_1,distance_2)
        lens_center = sphere.get_position() + distance * plane.get_normal()

        plane_int = plane.intersect(ray)
        if plane_int == np.inf:
            t1 = np.inf
        else:
            plane_pos_int = ray_position + plane_int * ray_dir 
            if np.linalg.norm(plane_pos_int - lens_center) <= self.get_lens_radius():
                t1 = plane_int
            else:
                t1 = np.inf

        sphere_int = sphere.intersect(ray)
        if sphere_int == np.inf:
            t2 = np.inf
        else:
            sphere_pos_int = ray_position + sphere_int * ray_dir 
            a = sphere_pos_int - plane.get_plane_position()
            if distance == distance_1:
                b = plane.get_normal(sphere_pos_int)
            else:
                b = -1* plane.get_normal(sphere_pos_int)
            if np.dot(a,b) > 0:
                t2= sphere_int
            else:
                t2 = np.inf
        if min(t1,t2) == np.inf:
            return np.inf    
        if min(t1,t2) == t1:
            self.__normal = plane.get_normal(plane_pos_int)
            return t1
        if min(t1,t2) == t2:
            self.__normal = sphere.get_normal(sphere_pos_int)
            return t2
    
    def get_normal(self,position):
        return self.__normal