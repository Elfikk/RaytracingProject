import numpy as np
import RayClass as rc
import networkx as nx
from ObjectClasses import Sphere, Plane
import matplotlib.pyplot as plt

a = 0
b = 0

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
        return 0.25 #0.1 to prevent clipping during rendering.

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
                    ray = ray, object = object, distance = distance - 1e-6, 
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
                    ray = rays[i], object = object, distance = distance - 1e-6, \
                        multiplier = multipliers[i])
                self.get_tree().add_edge(parent, next_node)
                next_node += 1

    def build_colour(self):
        nv = self.get_tree().nodes()
        #Have to make a list to be able to iterate with address along list.
        edges = list(self.get_tree().edges()) 
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
        if object.get_transmitivity():
            global a, b
            b += 1
            refracted_ray = ray.refracted_ray(object, distance)
            if type(refracted_ray) == rc.Ray:
                a += 1
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
                #print(self.get_screen())
                #print(i,j)
                screen_pos += inv_width * horizontal_line
                ray = rc.Ray(camera, screen_pos - camera)
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
        self.__samples = np.linspace(0.38, 0.75, samples + 2)[1:-1]

    def render(self, width = 400, height = 300, max_depth = 3,\
         dispersion_samples = 8):
        self.set_samples(dispersion_samples)
        image = Scene.render(self, width, height, max_depth)

        return image
    


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

    # objects = [Sphere(np.array([0, 0, 200]), 100, np.array([0.1,0.1,0.1]), 0.01, transmitivity=0.99), \
    #     Sphere(np.array([0, 0, 500]), 250, np.array([0.2,0.7,0.2]), 0.7), \
    #     Plane(np.array([0,1,-0.01]), np.array([0,-50,0]), np.array([0.7,0.2,0.2]), 0.9)]

    objects = [Sphere(np.array([200, 0, 700]), 200, np.array([0.9,0.7,0.9]), .1),\
        Sphere(np.array([0, 0, 200]), 100, np.array([0.1,0.1,0.1]), 0, transmitivity=1, refractive_index=1.1), \
            Plane(np.array([0,1,-0.01]), np.array([0,-200,0]), np.array([0.7,0.2,0.2]), 0.9)]

    #New method of specifying the viewport.
    viewport_corners = (np.array((-200, 150, 0)), np.array((200, 150, 0)),\
         np.array((-200,-150,0)))

    #Good practice is writing stuff earlier.
    camera_position = np.array((0,0,-10000))
    # camera_position = np.array([200,200,-500])
    light_pos = np.array((0,0,0)) #meaningless for now
    background_colour = np.array([0, 0, 0])

    scene = Scene(objects, camera_position, viewport_corners, light_pos,\
         background_colour)

    #Using default settings.
    #image = scene.render(800, 600, 3)
    image = scene.render(200, 150)

    #Funky test
    #image = scene.render(1600, 900)
    print(a,b)
    plt.imshow(image)
    plt.show()