import numpy as np
from RayClass import Ray
   
class SceneObject():
    #All scene objects inherit from this parent. 

    def __init__(self, colour, reflectivity = 1, transmitivity = 0, \
        refractive_index = 1):
        #Reflectivity is a multiplier for reflected ray intensities, and 
        #transmitivity for refracted rays. Can't create energy, 
        #so reflectivity + transmitivity <= 1.
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
        transmitivity = 0, refractive_index = 1, limits = [np.inf,np.inf]):
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

    def get_normal(self, position = np.array([0, 0, -1000])):
        #The default is very unlikely to lay on a plane. It's possible,
        # but we should be fine.
        orientation_vector = self.get_plane_position() - position
        orientation_vector = orientation_vector/np.linalg.norm(\
            orientation_vector)
        # print(self.__normal, orientation_vector)
        # print(np.sign(np.dot(orientation_vector, self.__normal)))
        #(np.dot(orientation_vector, self.__normal), np.sign(np.dot(orientation_vector, self.__normal)))
        return int(np.sign(np.dot(orientation_vector, self.__normal)))\
             * self.__normal

    # def get_normal(self, position = None):
    #     return self.__normal

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
        if self.get_limits() != [np.inf,np.inf]:
            point = ray_position + t + ray_dir
            if self.limits(point,self.get_limits()) == False:
                return np.inf
        return t
    
    def limits(self, point,lengths):
        position = self.get_plane_position()
        if np.abs(point[0] - position[0]) > lengths[1]/2:
            return False
        direction = np.cross([1,0,0],self.get_normal())
        length = (point[1] - position[1])/direction[1]
        if np.abs(length) > lengths[0]/2:
            return False
        return True

class Lens(SceneObject):

    def __init__(self, plane_position, plane_normal, sphere_position, sphere_radius,colour=[0,0,0], reflectivity = 0, \
        transmitivity = 1, refractive_index = 1.52):
        SceneObject.__init__(self, colour, reflectivity, transmitivity,\
            refractive_index)
        self.__plane = Plane(plane_position,plane_normal, colour,reflectivity = 0 )
        self.__sphere = Sphere(sphere_position,sphere_radius, colour,reflectivity = 0)
        self.__type = 'lens'

    def get_plane(self):
        return self.__plane

    def get_sphere(self):
        return self.__sphere

    def get_type(self):
        return self.__type

    def get_normal(self,position):
        return self.__normal
    
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
        rand_vector = plane.get_normal() + [1,0,0]
        plane_vector = np.cross(plane.get_normal(),rand_vector)
        ray_plane = Ray(lens_center,plane_vector)
        lens_radius = sphere.intersect(ray_plane)
        if lens_radius == np.inf:
            lens_radius =0

        if plane_int == np.inf:
            t1 = np.inf
        else:
            plane_pos_int = ray_position + plane_int * ray_dir 
            if np.linalg.norm(plane_pos_int - lens_center) <= lens_radius:
                t1 = plane_int
            else:
                t1 = np.inf

        sphere_int = sphere.intersect(ray)
        if sphere_int == np.inf:
            t2 = np.inf
        else:
            sphere_pos_int = ray_position + sphere_int * ray_dir 
            a = sphere_pos_int - plane.get_plane_position()
            if np.dot(a,plane.get_normal()) > 0:
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