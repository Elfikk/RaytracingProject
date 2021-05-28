class Ray():
#Class defining our funky rays.

    def __init__(self, pos_vector, dir_vector, wavelength = None, \
         intensity = 1):
        self.__position_vector = pos_vector
        self.__direction_vector = dir_vector
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

if __name__ == '__main__':
    ray = Ray([1,2,3], [3,2,1])
    print(ray.get_wavelength(), ray.get_position_vector(), \
        ray.get_direction_vector())
    ray2 = Ray((0,1,2), (1,1,1), 568)
    print(ray2.get_wavelength(), ray2.get_position_vector(), \
     ray2.get_direction_vector())