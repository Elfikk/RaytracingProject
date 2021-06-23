import numpy as np
import RayClass as rc
import networkx as nx
from ObjectClasses import Sphere, Plane, Circle
import matplotlib.pyplot as plt
import SellmeierCoefficients as sc
import RefractionMethods as rm

# a = 0
# b = 0
# c = 0

class Scene():
#Class used for holding the objects in the environment considered and 
#for handling ray process. Also holds the viewport, rays and light
#source.

    def __init__(self, objects, camera_position, screen_positions, \
        light_pos, background = np.array([0,0,0])):
        
        self.__objects = objects
        self.__camera_position = camera_position
        self.__light_pos = light_pos
        
        #Background colour
        self.__background = background

        #Three points, the top-left, top-right and bottom-left corner 
        #of the viewport. Allows for easy orientation change.
        self.__screen_positions = screen_positions
        self.__ray_tree = nx.Graph()

    def get_camera_position(self):
        return self.__camera_position

    def get_screen(self):
        return self.__screen

    def get_light_pos(self):
        return self.__light_pos

    def get_objects(self):
        return self.__objects

    def get_screen_positions(self):
        return self.__screen_positions

    def get_background(self):
        return self.__background

    def initiate_screen(self, width, height):
        self.__screen = np.zeros([width, height, 3])

    def update_screen(self, i, j, value):
        self.__screen[i, j] = value

    def get_screen(self):
        return self.__screen

    def get_tree(self):
        return self.__ray_tree

    def get_source_intensity(self, position):
        #Temporary until shadows and source of light implemented.
        #And that's more complicated than that anyway. So will
        #do for now.
        return 1 

    def nearest_intersection(self, ray):
        objects = self.get_objects()
        distances = []
        for object in objects:
            distances.append(object.intersect(ray))
        # print(distances)
        if np.min(distances) == np.inf:
            return None, np.inf
        else:
            i = distances.index(np.min(distances))
            return objects[i], distances[i]

    def pixel_colour(self, ray, max_depth):
        #The meat of the raytracer; this routine combines all the methods
        #involved in finding one pixel's colour.
        object, distance = self.nearest_intersection(ray)
        if distance == np.inf:
            return self.get_background()
        #print(distance)
        intersection = ray.get_position(distance)
        self.get_tree().add_node(0, intensity = \
                    self.get_source_intensity(intersection), colour = \
                    object.get_colour(), running_colour = np.array([0,0,0]), \
                    ray = ray, object = object, distance = distance, 
                    multiplier = 1)
        self.form_tree(max_depth)
        return self.build_colour()

    def form_tree(self, max_depth):
        
        #Tracks whether new nodes were added after the last call.
        changes = True 
        #Keeps track of the depth of the tree.
        count = 0 
        
        #The last node in the depth above. -1 value initially to get root
        #in loop -1 + 1 = 0.
        last_node = -1 

        #nv is global, so does not need to be called multiple times.
        #I suspect it stores the reference to the environment as opposed to
        #the data itself.
        nv = self.get_tree().nodes() 

        while count < max_depth and changes == True:

            #No changes yet.
            changes = False
            #Incremement depth.
            count += 1

            for i in range(last_node+1, len(self.get_tree())):
                ray, object, distance = nv[i]['ray'], nv[i]['object'],\
                     nv[i]['distance']
                rays, multipliers = self.intersection_rays(ray, object,\
                     distance)
                if len(rays) != 0:
                    self.form_children(rays, i, multipliers)
                    #Children rays have been added.
                    changes = True

    def form_children(self, rays, parent, multipliers):
        #parent is the id of the (parent) node above.

        #We name nodes by ID numbers in order in which they were added.
        next_node = max(self.get_tree().nodes) + 1
        
        #Currently loops over two rays, but remember, modified dispersion
        #will require more!
        for i in range(len(rays)):
            object, distance = self.nearest_intersection(rays[i])
            if distance != np.inf:
                intersection = rays[i].get_position(distance)
                self.get_tree().add_node(next_node, intensity = \
                    self.get_source_intensity(intersection), colour = \
                    np.array(object.get_colour()), running_colour = np.array([0,0,0]), \
                    ray = rays[i], object = object, distance = distance, \
                        multiplier = multipliers[i])
                self.get_tree().add_edge(parent, next_node)
                next_node += 1

    def build_colour(self):
        nv = self.get_tree().nodes()
        #Have to make a list to be able to iterate with address along list.
        edges = list(self.get_tree().edges()) 
        #print(edges)
        for i in range(len(edges)):
            next_edge = edges[-i - 1]
            parent, child = next_edge
            child_running, colour, intensity, multiplier = nv[child][ \
                'running_colour'], nv[child]['colour'], nv[child]['intensity'],\
                     nv[child]['multiplier']
            parent_running = nv[parent]['running_colour']
            #Colour blending test (actually works really nicely)
            # total_colour = (multiplier * (child_running + colour * intensity)\
            #      + parent_running)/2
            total_colour = multiplier * (child_running + colour * intensity)\
                  + parent_running
            nv[parent]['running_colour'] = total_colour
        # final_colour = nv[0]['multiplier'] * (nv[0]['running_colour'] + \
        #     nv[0]['colour'] * nv[0]['intensity'])
        final_colour = nv[0]['running_colour'] + nv[0]['multiplier'] * \
            (nv[0]['colour'] * nv[0]['intensity'])
        self.get_tree().clear()
        return final_colour

    def intersection_rays(self, ray, object, distance):
        #Method for finding the reflected and refracted rays off an object.
        rays = []
        #Takes the multiplier of the object it reflected/refracted off,
        #so we don't need to track the types of rays used later.
        multipliers = [] 
        # global a, b, c
        # c += 1
        if object.get_transmitivity():
            # b += 1
            refracted_ray = ray.refracted_ray(object, distance)
            if type(refracted_ray) == rc.Ray:
                # a += 1
                rays.append(refracted_ray)
                multipliers.append(object.get_transmitivity())
        if object.get_reflectivity():
            rays.append(ray.reflected_ray(object, distance))
            multipliers.append(object.get_reflectivity())
        return rays, multipliers

    def render(self, width = 400, height = 300, max_depth = 3):
        #Width/Height Independent of Screen dimensions; can go wrong if 
        #you mess up your aspect ratio i.e spheres can look like ellipses,
        #but this allows us to disconnect the dependence of the dimensions
        #of the viewport and its resolution.
        
        self.initiate_screen(height, width)

        #top-left, top-right, bottom-left corners.
        tl, tr, bl = self.get_screen_positions()
        camera = self.get_camera_position()

        horizontal_line = tr - tl
        vertical_line = bl - tl

        inv_height = 1/height #Inverse Height
        inv_width = 1/width #Inverse Width

        # print(horizontal_line, vertical_line)
        # print(inv_width* horizontal_line, inv_height * vertical_line)

        #shape = np.shape(self.get_screen())
        #print(shape)

        for j in range(height):
            screen_pos = tl + (j + 0.5)* inv_height * vertical_line \
                + 0.5 * inv_width * horizontal_line
            print(j) #A poor man's progress bar.
            for i in range(width):
                #print(j,i)
                #print(self.get_screen())
                #print(i,j)
                screen_pos += inv_width * horizontal_line
                ray = rc.Ray(camera, screen_pos - camera)
                #ray = rc.Ray(screen_pos, np.array([0,0,1])) #Paraxial rays.
                self.update_screen(j,i, self.pixel_colour(ray, max_depth))
        
        return self.get_screen()

class Dispersion_Scene(Scene):
    #This Scene sub-class overwrites some methods used in normal rendering.
    #The idea is to reduce the number of ifs involved in rendering, to not
    #make it even slower.

    def __init__(self, objects, camera_position, screen_positions, \
        light_pos, background = np.array([0,0,0])):

        Scene.__init__(self, objects, camera_position, screen_positions, \
        light_pos, background)
        self.__samples = np.array([0.565])

    def set_samples(self, samples):
        #Samples in the optical wavelength range.
        self.__samples = np.linspace(0.38, 0.75, samples + 2)[1:-1]

    def get_samples(self):
        return self.__samples

    def render(self, width = 400, height = 300, max_depth = 3,\
        dispersion_samples = 8):
        self.set_samples(dispersion_samples)
        print(self.get_samples())
        image = Scene.render(self, width, height, max_depth)

        return image

    def intersection_rays(self, ray, object, distance):
        #Method for finding the reflected and refracted rays off an object.
        rays = []
        #Takes the multiplier of the object it reflected/refracted off,
        #so we don't need to track the types of rays used later.
        multipliers = [] 
        if object.get_transmitivity():
            if ray.get_wavelength() == None:
                wavelengths = self.get_samples()
                sample_number = len(wavelengths)
                for i in range(len(wavelengths)):
                    refracted_ray = ray.wavelength_refraction(object, distance, \
                        wavelengths[i])
                    if type(refracted_ray) == rc.Ray:
                        rays.append(refracted_ray)
                        multipliers.append(object.get_transmitivity()/sample_number)
            else:
                refracted_ray = ray.wavelength_refraction(object, distance, \
                        ray.get_wavelength())
                if type(refracted_ray) == rc.Ray:
                    rays.append(refracted_ray)
                    multipliers.append(object.get_transmitivity())
        if object.get_reflectivity():
            rays.append(ray.reflected_ray(object, distance, ray.get_wavelength()))
            multipliers.append(object.get_reflectivity())
        return rays, multipliers
    
    def form_children(self, rays, parent, multipliers):
        #parent is the id of the (parent) node above.

        wavelengths = {}
        number_of_samples = len(self.get_samples())
        for wavelength in self.get_samples():
            wavelengths[wavelength] = 0

        #We name nodes by ID numbers in order in which they were added.
        next_node = max(self.get_tree().nodes) + 1

        #Currently loops over two rays, but remember, modified dispersion
        #will require more!
        for i in range(len(rays)):
            object, distance = self.nearest_intersection(rays[i])
            if distance != np.inf:
                #So that it doesn't refer to the same dictionary (python at its finest)
                ray_wavelengths = wavelengths.copy() 
                intersection = rays[i].get_position(distance)
                if rays[i].get_wavelength() == None:
                    for wavelength in ray_wavelengths: #Iterates over keys
                        ray_wavelengths[wavelength] += multipliers[i]/number_of_samples
                    self.get_tree().add_node(next_node, intensity = \
                        self.get_source_intensity(intersection), wavelengths =\
                        ray_wavelengths, ray = rays[i], object = object, \
                        distance = distance)
                    self.get_tree().add_edge(parent, next_node)
                else:
                    ray_wavelengths[rays[i].get_wavelength()] = multipliers[i]# \
                        # * rm.wavelength_correlation(rays[i].get_wavelength(),\
                        #      object.get_colour())
                    self.get_tree().add_node(next_node, intensity = \
                        self.get_source_intensity(intersection), wavelengths =\
                        ray_wavelengths, ray = rays[i], object = object, \
                        distance = distance)
                    self.get_tree().add_edge(parent, next_node)
                next_node += 1

    def pixel_colour(self, ray, max_depth):
        #The meat of the raytracer; this routine combines all the methods
        #involved in finding one pixel's colour.
        object, distance = self.nearest_intersection(ray)
        if distance == np.inf:
            return self.get_background()
        #print(distance)
        wavelengths = self.get_samples()
        sample_number = len(wavelengths)
        ray_wavelengths = {}
        for wavelength in wavelengths:
            ray_wavelengths[wavelength] = 1/sample_number
        intersection = ray.get_position(distance)
        self.get_tree().add_node(0, intensity = \
                    self.get_source_intensity(intersection), wavelengths = \
                    ray_wavelengths, ray = ray, object = object, \
                    distance = distance)
        self.form_tree(max_depth)
        return self.build_colour()

    def build_colour(self):
        nv = self.get_tree().nodes()
        #Have to make a list to be able to iterate with address along list.
        edges = np.array(list(self.get_tree().edges()))
        #print(edges)
        #for i in range(len(edges)):

        wavelengths = {}
        #number_of_samples = len(self.get_samples())
        for wavelength in self.get_samples():
            wavelengths[wavelength] = 0

        while len(edges) != 0:
            
            next_edge = edges[-1]
            parent = next_edge[0]
            #Cursed line, I hate numpy syntax
            children_addresses = np.where(edges[:,0] == parent) 
            childrens = edges[children_addresses] #As Howard would say.
            children_wavelengths = wavelengths.copy()
            #intensities = np.zeros(number_of_samples)
            for i in range(len(childrens)):
                for wavelength in nv[childrens[i][1]]['wavelengths']:
                    children_wavelengths[wavelength] += \
                        nv[childrens[i][1]]['wavelengths'][wavelength]

            nv[parent]['wavelengths'] = children_wavelengths.copy()
            edges = np.delete(edges, children_addresses, 0)

        final_wavelengths = nv[0]['wavelengths'] # oh boi
        intensity_total = sum(final_wavelengths.values())
        #print(final_wavelengths)
        final_colour = np.array([0.,0.,0.])
        for wavelength in final_wavelengths:
            # print(wavelength)
            # print(final_wavelengths[wavelength])
            final_colour += final_wavelengths[wavelength] * \
                rm.wavelength_rgb(wavelength)
        self.get_tree().clear()
        return final_colour        

class Rainbow_Tracer():
#Uses raytracing idea, but doesn't explicitely use our raytracer as it 
#needs some heavy editing - so I'd rather just write it without inheritance
#causing possible trouble.

#It's still a raytraced rainbow (hopefully as of writing), just doesn't use
#the exact same method as the previous scenes.

    def __init__(self, camera_position, screen_positions, background = \
        np.array([0.4, 0.6, 0.8])):

        #These guys are used just as they were earlier
        self.__camera_position = camera_position
        self.__screen_positions = screen_positions
        self.__background = background 

        #Raindrop modelled as a sphere of radius of 1mm (real raindrops are
        #not spherical, but look like hamburger buns or parachutes).
        self.__raindrop = Sphere(np.array([0,0,0]), 1e-3, np.array([1,1,1]),\
            reflectivity= 0, transmitivity = 1, sellmeier_Bs= sc.water_Bs, \
            sellmeier_Cs= sc.water_Cs) 

        #Currently assumes scene in the positive z direction.
        sun_angle = np.deg2rad(30)
        self.__light_normal = np.array([0, np.sin(sun_angle), -np.cos(sun_angle)])

        self.__wavelengths = np.array([0.565])

    def get_wavelengths(self):
        return self.__wavelengths

    def render(self, width = 600, height = 300, samples = 8, rolls = 10):

        image = np.zeros((height, width, 3))

        self.__wavelengths = np.linspace(0.38, 0.75, samples + 2)[1:-1]

        tl, tr, bl = self.__screen_positions 
        camera = self.__camera_position

        horizontal_line = tr - tl
        vertical_line = bl - tl

        inv_height = 1/height #Inverse Height
        inv_width = 1/width #Inverse Width

        for j in range(height):
            screen_pos = tl + (j + 0.5)* inv_height * vertical_line \
                + 0.5 * inv_width * horizontal_line
            print(j) #A poor man's progress bar.
            for i in range(width):
                #print(j,i)
                #print(self.get_screen())
                #print(i,j)
                screen_pos += inv_width * horizontal_line
                ray = rc.Ray(screen_pos, screen_pos - camera)
                #ray = rc.Ray(screen_pos, np.array([0,0,1])) #Paraxial rays.
                image[j, i] = self.pixel_colour(ray, rolls)
        
        return image

    def pixel_colour(self, ray, rolls):
        #Order of magnitude estimate gives bout 0.06m of seperation between raindrops.
        #So I use 0.1 because I'm a big fan of nicer numbers.

        # distances = np.arange(5, 20+layer_distance, layer_distance)/ray.get_direction_vector()[2]
        # intersections = ray_distances(ray, distances, layer_distance)
        # if len(intersections) > 1:
        #     return self.raindrop_effect(ray, distances[intersections[0]], layer_distance)
        # return self.__background

        count = 0
        centre = ray.get_position(.1) 
        dot_averages = np.zeros(len(self.__wavelengths))

        while count < rolls:
            #I "spawn" the raindrop 1 metre away on the ray path. Arbitrary 
            #choice as we only care about angles. However, picking a smaller
            #distance like a metre 
            distance1 = np.inf #random distance from the 'centre'.
            dots = np.array([])
            while distance1 == np.inf:
                #I want samples from -2 mm to 2mm on each axis, with the 
                #distance constraint - as we do want our ray to intersect with
                #the raindrop after all!
                pos_change = -2e-3 + 4e-3 * np.array([np.random.random(), np.random.random(), np.random.random()])
                self.__raindrop.set_centre(centre + pos_change)
                distance1 = self.__raindrop.intersect(ray) 
            for wavelength in self.__wavelengths:
                dots = np.append(dots, self.raindrop_bouncing(ray, distance1, wavelength))
            dot_averages += dots/rolls
            count += 1

        print(dot_averages)

        index = np.where(np.logical_and(dot_averages > 0., dot_averages == max(dot_averages)))
        try:
            return rm.wavelength_rgb(self.__wavelengths[index[0][0]])
        except:
            return self.__background

    # def raindrop_centre(self, centre):
        

    def raindrop_effect(self, ray, distance, layer_distance):
        
        global a
        a += 1

        sphere_centre = ray.get_position(distance) - ray.get_position(distance)%layer_distance \
            + np.array([layer_distance/2, layer_distance/2, layer_distance/2])
        self.raindrop_centre(sphere_centre)
        wavelengths = self.get_wavelengths()
        
        #The colour of the pixel is determined by maximising the dot product
        #of the ray against the Sun plane. The wavelength which has maximum 
        #dot product has the smallest angle to the incoming rays of the Sun,
        #so will be most intense.
        dot_products = np.array([])

        #Distance along ray line for the first intersection.
        distance1 = self.__raindrop.intersect(ray) 

        for wavelength in wavelengths:
            dot = self.raindrop_bouncing(ray, distance1, wavelength)
            dot_products = np.append(dot_products, dot)

        max_dot_index = np.where(np.logical_and(dot_products > 0.85, \
            dot_products == np.amax(dot_products)))

        if len(max_dot_index) != 0:
            return rm.wavelength_rgb(wavelengths[max_dot_index[0]])
        return self.__background

    def raindrop_bouncing(self, ray, distance1, wavelength):
        refracted_ray = ray.wavelength_refraction(self.__raindrop,\
                distance1, wavelength)
        #Distance travelled inside the raindrop to hit it a second time.
        distance2 = self.__raindrop.intersect(refracted_ray) 
        if distance2 != np.inf:
            #Is expected to be a tir ray, but could be refracted if not
            #beyond critical angle.
            tir_ray = refracted_ray.wavelength_refraction(self.__raindrop,\
            distance2, wavelength)
            distance3 = self.__raindrop.intersect(tir_ray)
            if distance3 != np.inf:
                final_ray = tir_ray.wavelength_refraction(self.__raindrop,\
                distance3, wavelength)
                return np.dot(final_ray.get_direction_vector(), \
                    self.__light_normal)
            return -1
        return -1

def ray_distances(ray, t, layer_distance = 0.1):
    #Works on an array of ts, giving much faster calculation.

    #Distances along the ray needed to pass at every 0.1 increment along z.

    x = (ray.get_position_vector()[0] + t * ray.get_direction_vector()[0])%layer_distance
    y = (ray.get_position_vector()[1] + t * ray.get_direction_vector()[1])%layer_distance
    
    #Distance between ray in 2d plane and 
    distance_squared = (x - layer_distance/2)**2  + (y - layer_distance/2)**2
    return np.where(distance_squared < 4e-6)

if __name__ == '__main__':
    
    #Testing whether rendering works, before changing main :)

    #Objects as per usual, nothing changed here. I am allowing this one
    #PEP8 violation :)
    # objects = [Plane(np.array([-50,0,-10]), np.array([0,-220,0]), np.array([0.7,0.2,0.2]), 0.2), \
    #        Sphere(np.array([-300,50,200]), 100, np.array([0.2,0.7,0.2]), 0.7), \
    #        Sphere(np.array([-100,100,200]), 25, np.array([0.2,0.2,0.7]), 0.5)]

    # objects = [Plane(np.array([0,1,-0.01]), np.array([0,-50,0]), np.array([0.7,0.2,0.2]), 0.9), \
    #        Sphere(np.array([0, 0, 500]), 50, np.array([0.2,0.7,0.2]), 0.2), \
    #        Plane(np.array([1,0,-1]), np.array([0,0,125]), np.array([0.2,0.2,0.2]), \
    #             transmitivity = 0.9, refractive_index = 1.5, reflectivity=0.1)]
    
    # objects = [Plane(np.array([0,1,0]), np.array([0,-100,0]), np.array([0.7,0.2,0.2]), 0.3), \
    #        Sphere(np.array([0, 50, 500]), 75, np.array([0.2,0.7,0.2]), .9), \
    #        Plane(np.array([0,0,10]),np.array([0,0,300]), np.array([0.1,0.1,0.1]), transmitivity = 0.9,\
    #        refractive_index = 1.5, reflectivity= 0.1)]

    # objects = [Plane(np.array([0,1,-0.01]), np.array([0,-200,0]), np.array([0.7,0.2,0.2]), 0.9)]

    # objects = [Sphere(np.array([150, 0, 250]), 100, np.array([0.2,0.7,0.2]), 0.7)]

    # objects = [Sphere(np.array([0, 0, 200]), 100, np.array([0.1,0.1,0.1]), 0.01, transmitivity=0.99,\
    #     sellmeier_Bs=sc.crown_glass_Bs, sellmeier_Cs=sc.crown_glass_Cs), \
    #     Sphere(np.array([0, 0, 500]), 250, np.array([0.2,0.7,0.2]), 0.7), \
    #     Plane(np.array([0,1,-0.01]), np.array([0,-50,0]), np.array([0.7,0.2,0.2]), 0.9)]

    # objects = [Sphere(np.array([100, 150, 1000]), 200, np.array([0.9,0.7,0.9]), 0.5, transmitivity=0, refractive_index=1.5),\
    #     Sphere(np.array([0, 0, 300]), 150, np.array([0.1,0.1,0.1]), 0.1, transmitivity=.9, refractive_index=1.5), \
    #     Sphere(np.array([-200, -75, 600]), 100, np.array([0.5,0.5,0.8]), 0.5, transmitivity=0, refractive_index=1.5)]
    # #     #    Plane(np.array([0,1,-0.01]), np.array([0,-200,0]), np.array([0.7,0.2,0.2]), 0.9)]

    # # objects = [Sphere(np.array([100, 150, 1000]), 200, np.array([0.9,0.7,0.9]), 0.5, transmitivity=0, refractive_index=1.5),\
    #     Sphere(np.array([0, 0, 300]), 150, np.array([0.1,0.1,0.1]), 0.1, transmitivity=.9, refractive_index=1.01), \
    #     Sphere(np.array([-200, -75, 600]), 100, np.array([0.5,0.5,0.8]), 0.5, transmitivity=0, refractive_index=1.5)]
    # #     #    Plane(np.array([0,1,-0.01]), np.array([0,-200,0]), np.array([0.7,0.2,0.2]), 0.9)]

    objects = [Sphere(np.array([200, 150, 3000]), 400, np.array([1.,1.,1.]), 0.),\
        # Sphere(np.array([0, 0, 300]), 150, np.array([0.,0.,0.]), 0., transmitivity=1,\
        #      sellmeier_Bs=sc.flint_glass_Bs, sellmeier_Cs=sc.flint_glass_Cs, refractive_index=1.6)]#, \
        Plane(np.array([0,0,1]), np.array([0,0,300]), np.array([0.,0.,0.]), 0, transmitivity=1, \
            sellmeier_Bs=sc.flint_glass_Bs, sellmeier_Cs=sc.flint_glass_Cs)]
    
    #     # Sphere(np.array([-200, -75, 600]), 100, np.array([0.5,0.5,0.8]), 0.5, transmitivity=0, \
        #     refractive_index=1.5)]

    # objects = [Circle(np.array([1,0,-0.1]), np.array([150,0,0]), np.array([0.1, 0.5, 0.8]), radius=50)]

    #New method of specifying the viewport.
    viewport_corners = (np.array((-200, 150, 0)), np.array((200, 150, 0)),\
         np.array((-200,-150,0)))

    # viewport_corners = (np.array((-55, 20, 0)), np.array((-40, 20, 0)),\
    #      np.array((-55,-30,0)))

    # viewport_corners = (np.array((-400, 300, 0)), np.array((400, 300, 0)),\
    #      np.array((-400,-300,0)))

    # viewport_corners = (np.array((-1, 1, 0)), np.array((1, 1, 0)),\
    #      np.array((-1,-1,0)))

    # viewport_corners = (np.array((-46, -7.5, 0)), np.array((-44, -7.5, 0)),\
    #       np.array((-46,-12.5,0)))

    # viewport_corners = (np.array((13.5, -7.5, 0)), np.array((16.5, -7.5, 0)),\
    #      np.array((13.5,-15,0)))

    # viewport_corners = (np.array((10, 30, 0)), np.array((30, 30, 0)),\
    #       np.array((10,10,0)))

    # viewport_corners = (np.array((10, 28.15, 0)), np.array((11.5, 28.15, 0)),\
    #       np.array((10,27.5,0)))

    # # #Good practice is writing stuff earlier.
    # camera_position = np.array((0,0,-250))
    # # camera_position = np.array([200,200,-500])
    # light_pos = np.array((0,0,0)) #meaningless for now
    # background_colour = np.array([0, 0, 0])

    # scene = Dispersion_Scene(objects, camera_position, viewport_corners, light_pos,\
    #      background_colour)

    # scene = Scene(objects, camera_position, viewport_corners, light_pos,\
    #       background_colour)

    #Using default settings.
    #image = scene.render(800, 600, 3)
    #image = scene.render(2, 2, 3)

    #image = scene.render(75, 250, 2)

    #image = scene.render(400, 300, 3, 4)
    #image = scene.render(2,2,2,2)
    #image = scene.render(1600, 1200, 5)
    #image = scene.render(200, 500, 2)
    #image = scene.render(50, 50, 2)
    #image = scene.render(20, 100, 3, 16)
    #image = scene.render(30, 75, max_depth=2, dispersion_samples= 6)
    #image = scene.render(100, 100, max_depth=2, dispersion_samples=16)

    #image = scene.render(900, 390, 2, 16)

    # image = scene.render(8, 20, 2)
    # normalisation = np.amax(image)
    # image = image / normalisation #to fix stupid clipping and intensity :)

    #print(a,b,c, rc.d, rc.e, rc.f) #Data gathering.

    #Rainbow

    camera_position = np.array([0,0,-5])

    human_angle = np.deg2rad(60)

    viewport_corners = (np.array([-5 * np.tan(human_angle), 5 * np.tan(human_angle), 0]),\
                        np.array([5 * np.tan(human_angle),5 * np.tan(human_angle),0]),\
                        np.array([-5 * np.tan(human_angle), 0, 0]))

    scene = Rainbow_Tracer(camera_position, viewport_corners)

    image = scene.render(50, 25, 8, 20)

    plt.imshow(image)
    plt.show()