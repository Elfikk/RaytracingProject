import numpy as np
# import SellmeierCoefficients as sc
import matplotlib.pyplot as plt

def rgb_decorator(function):
    #Wavelengths will be sampled exactly the same way for each secondary
    #ray set. By storing the results of rgb conversion, we will waste less
    #computation time.
    cache = {}
    def wrapper(*args, **kwargs):
        if args in cache:
            return cache[args]
        cache[args] = function(*args,**kwargs)
        return cache[args]
    
    return wrapper

def attenuation_decorator(function):
    #Same exact thing, but used for the attenuation functions.
    cache = {}
    def wrapper(wavelength):
        if wavelength in cache:
            return cache[wavelength]
        cache[wavelength] = function(wavelength)
        #print('a') just for checking it works (it does)
        return cache[wavelength]

    return wrapper

def refracted_direction(incident, normal, n_2, n_1):
    #Returns the direction vector of the secondary ray.
    #Normal is not antiparallel to the ray direction's parallel component.
    #Expects normalised vectors!
    mu = n_1/n_2
    parallel = np.sqrt(1 - mu**2 * (1 - (np.dot(incident,normal)**2)))\
             * normal
    perpendicular = mu * (incident - np.dot(incident, normal)*normal )
    return (parallel + perpendicular)/np.linalg.norm(parallel + perpendicular)

def refracted_ray(intersection, refracted_dir):
    return Ray(intersection, refracted_dir) 

def sellmeier_n(wavelength, Bs, Cs):
    #COEFFICIENTS MUST BE PASSED AS A NUMPY ARRAY.
    #Wavelength is in μm, Cs are μm^2 and Bs dimensionless.
    tops = wavelength**2 * Bs
    bots = wavelength**2 - Cs
    terms = tops/bots
    return np.sqrt(1 + sum(terms))

@rgb_decorator
def wavelength_rgb(wavelength, gamma = 0.8):
    #Takes a wavelength in μm and returns an rgb value for it.
    #Based off an approximation. Real rgb considerations wee bit 
    #complicated; too much for something that requires quick 
    #evaluation.
    if isinstance(wavelength, float):
        return np.array([red(wavelength,gamma), green(wavelength,gamma), \
            blue(wavelength,gamma)])
    return np.array((1,1,1)) #None type returns white colour (ray made of all colours)

def red(wavelength, gamma):
    #Red component. A lot of elifs sadly but ye.
    if 0.38 < wavelength < 0.44:
        return ((0.44-wavelength)/0.06 * attenuation_short(wavelength))**gamma
    elif 0.51 < wavelength < 0.58:
        return ((wavelength - 0.51)/0.07)**gamma
    elif 0.58 <= wavelength <= 0.645:
        return 1
    elif 0.645 < wavelength < 0.75:
        return attenuation_long(wavelength)**gamma
    return 0

def green(wavelength, gamma):
    #Same goes for green.
    if 0.44 < wavelength < 0.49:
        return ((wavelength - 0.44)/0.05)**gamma
    elif 0.49 <= wavelength <= 0.58:
        return 1
    elif 0.58 < wavelength < 0.645:
        return ((0.645 - wavelength)/0.065)**gamma
    return 0

def blue(wavelength, gamma):
    #Bloo passport.
    if 0.38 < wavelength < 0.44:
        return attenuation_short(wavelength)**gamma
    elif 0.44 <= wavelength <= 0.49:
        return 1
    elif 0.49 < wavelength < 0.51:
        return ((0.51 - wavelength)/0.02)
    return 0

@attenuation_decorator
def attenuation_short(wavelength):
    return 0.3 + 0.7*((wavelength-0.38)/0.06)

@attenuation_decorator
def attenuation_long(wavelength):
    return 0.3 + 0.7*((0.75-wavelength)/0.105)

def correlation(colour1, colour2):
    #BTEC Absorption.
    #Max magnitude is 3, but want values between 0 and 1
    multiplier = np.dot(colour1, colour2)/3 
    return multiplier

def wavelength_correlation(wavelength, object_colour):
    ray_colour = wavelength_rgb(wavelength)
    return correlation(ray_colour, object_colour)

if __name__ == '__main__':
    
    #Rays moving through the same medium should not be refracted (should
    #be the case with arbitrary normal)
    i = np.array([1,0,0])
    n = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    n1 = 1 ; n2 = 1
    print(refracted_direction(i, n, n2, n1)) #Passed

    #Rays moving at the boundary (parallel to the surface) should
    #not be affected (well, we don't consider diffraction!).
    n = np.array([0, 1, 0])
    print(refracted_direction(i, n, n2, n1)) #Passed

    #Rays along the normal should not be affected.
    print(refracted_direction(i, i, n2, n1)) #Passed

    #Worked example. Ray at 45 angle moving into plane with refractive
    #index 2. Expected = [sqrt(2)/4, sqrt(14)/4, 0] = [0.35..., 0.93..., 0]
    i = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    n = np.array([1,0,0])
    n2 = 2
    print(refracted_direction(i, n, n2, n1)) #Passed

    #Just checking if it explodes:
    i = np.array([1, 1, 1]) / np.sqrt(3)
    n = np.array([0, 1, 1]) / np.sqrt(2)
    a = refracted_direction(i, n, n2, n1) 
    print(a, np.linalg.norm(a)) #Doesn't explode. Rounding error visible.

    #What happens for n = 0? Zero division error for n2 = 0, and n1 = 0
    #returns the normal of the surface - as would be predicted by Snell's
    #Law actually.
    print(refracted_direction(i, n, 1, 0))

    #What about negative indices? Metamaterials right? Gibberish that 
    #would need to get tested inside the program itself. At least it doesn't
    #crash. Would be a cool way to expand on the project though no? 
    print(refracted_direction(i, n, 1, -1))

    #TIR. How does that look like?
    i = np.array([1,0,0]) 
    n = np.array([-np.sin(np.pi/3), -np.cos(np.pi/3), 0])
    print(refracted_direction(i, n, 1, 5))
    #Returns [nan, nan, nan] - something to watch out for. Perhaps check
    #before call whether TIR occurs.

    #Testing importing of constants from other files.
    # Bs = cs.water_Bs
    # print(Bs)

    # R = wavelength_rgb(0.66, 1)
    # print(R)

    # a = wavelength_rgb(0.66, 1)
    # print(a)

    # numero = 1000
    # samples = np.linspace(0.38, 0.75, numero)[1:-1]
    # colours = np.zeros((numero-2,numero-2, 3))
    # for i in range(len(samples)):
    #     colours[i] = wavelength_rgb(samples[i]) 
    # print(255 * sum(colours)/(numero-2)) #RGB values don't average at white.
    # plt.imshow(colours)
    # plt.show()

    #print(wavelength_rgb(None))
    # samples = np.linspace(0.38, 0.75, 102)[1:-1]
    # correlation_list = []
    # for i in range(len(samples)):
    #     correlation_list.append(wavelength_correlation(samples[i], \
    #         np.array([1, 0, 0])))
    # correlation_samples = np.array(correlation_list)
    # plt.plot(samples, correlation_samples)
    # plt.show()

    

    # n = sellmeier_n(0.565, sc.flint_glass_Bs, sc.flint_glass_Cs)
    # print(n)
    