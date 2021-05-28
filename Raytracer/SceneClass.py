from numpy import array
import RayClass as rc

class Scene():
#Class used for holding the objects in the environment considered and 
#for handling ray process. Also holds the viewport, rays and light
#source.

    def __init__(self, objects, camera_position, screen_position, \
        screen_width, screen_height, light_pos):
        
        self.__objects = objects
        self.__camera_position = camera_position
        self.__light_pos = light_pos

        #Debating on where to define this. Either top-left or centre.
        self.__screen_position = screen_position 
        
        #Also - currently assuming the screen plane is vertical - but
        #maybe adding tilt wouldn't be such a bad idea? For the sake of
        #having the option in the long term.
        self.__screen = array((screen_width, screen_height, 3))

        #Currently working in a list, but could be changed.
        self.__rays = []
    
    def fire_camera_ray(self, screen_y, screen_z):
        #Currently assumes screen is  vertical screen. 
        #Can be reworked if needed. 
        cam_pos = self.get_camera_position()
        dir_vector = array([0, screen_y, screen_z]) 
        new_ray = rc.Ray(cam_pos, dir_vector)
        self.add_ray(new_ray)

    def add_ray(self, ray):
        self.__rays.append(ray)

    def get_camera_position(self):
        return self.__camera_position

    def get_screen(self):
        return self.__screen

    def get_light_pos(self):
        return self.__light_pos

    def get_objects(self):
        return self.__objects

    def get_screen_position(self):
        return self.__screen_position

    def update_screen(self, array_position, value):
        self.get_screen()[array_position] += value

    def clear_rays(self):
        #When finished with a set of rays, clear the list.
        self.__rays = []

    