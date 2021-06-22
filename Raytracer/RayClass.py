import numpy as np
import RefractionMethods as rm
import networkx as nx

# d = 0
# e = 0
# f = 0

class Ray():
#Class defining our funky rays.

    epsilon = 1e-6 #The small point correction used everywhere.

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
    
    #Needed it for a bit, then changed code again.
    def set_wavelength(self, wavelength): 
        self.__wavelength = wavelength

    #I retract the secondary ray preference comments, worked well for 
    #me today :p
    def reflected_ray(self, object, distance, wavelength = None):
        dir = self.get_direction_vector()
        reflected_pos = self.get_position(distance)
        normal = object.get_normal(reflected_pos)

        #Changed it whilst seeing if I it could be causing an error - it
        #didn't, but functionality is the same so I'm leaving it.
        normal_component = np.dot(dir, normal) * normal
        plane_component = dir - normal_component
        
        reflected_dir = plane_component - normal_component
        reflected_dir = reflected_dir/np.linalg.norm(reflected_dir) 

        reflected_ray = Ray(reflected_pos + Ray.epsilon * reflected_dir,\
             reflected_dir, wavelength)

        return reflected_ray

    def refracted_ray(self, object, distance):
        #Doesn't take external refractive indices; don't put objects into each
        #other or at each other's boundaries (tiny limitation), but due to n's
        #being associated with boundary, not environments. Can put this as a
        #possible improvement at the end of the project (or if needed, fix it).
        n_air = 1
        n_obj = object.get_refractive_index()
        intersection = self.get_position(distance)
        normal = object.get_normal(intersection)
        dir_vector = self.get_direction_vector()
        if np.sign(np.dot(dir_vector,normal)) == 1.0: 
            direction = rm.refracted_direction(self.get_direction_vector(), \
                normal, n_air, n_obj)
            if not np.isnan(direction[0]):
                return Ray(intersection + Ray.epsilon * direction, direction)
            return self.reflected_ray(object, distance)
        direction = rm.refracted_direction(self.get_direction_vector(), \
            - normal, n_obj, n_air)
        if np.dot(np.cross(direction, normal), np.cross(direction, normal))\
             > n_air / n_obj**2:
             return None
        return Ray(intersection + Ray.epsilon * direction, direction)

    def wavelength_refraction(self, object, distance, wavelength):
        n_air = 1
        n_obj = object.functional_n(wavelength)
        intersection = self.get_position(distance)
        normal = object.get_normal(intersection)
        dir_vector = self.get_direction_vector()
        if np.sign(np.dot(dir_vector,normal)) == 1.0: 
            direction = rm.refracted_direction(self.get_direction_vector(), \
                normal, n_air, n_obj)
            if not np.isnan(direction[0]):
                return Ray(intersection + Ray.epsilon * direction, direction, \
                    wavelength)
            tir_ray = self.reflected_ray(object, distance, wavelength)
            return tir_ray
        direction = rm.refracted_direction(self.get_direction_vector(), \
            - normal, n_obj, n_air)
        if np.dot(np.cross(direction, normal), np.cross(direction, normal))\
             > n_air / n_obj**2:
             return None
        return Ray(intersection + Ray.epsilon * direction, direction, wavelength)
        
if __name__ == '__main__':

#     #Ray can be used for identifying nodesssss, although not how 
#     #implemented in the end.
#     ray = Ray(np.array([1,1,1]), np.array([1,1,1]))
#     graph = nx.Graph()
#     graph.add_node(ray)

    rays = []
    wavelengths = np.linspace(0.38, 0.75, 10)[1:-1]
    for i in range(len(wavelengths)):
        rays.append(Ray(np.array([1,1,1]), np.array([1,1,1]), wavelength=wavelengths[i]))
    print(rays)