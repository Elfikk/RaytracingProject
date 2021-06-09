import numpy as np

def refracted_direction(incident, normal, n_2, n_1):
    #Returns the direction vector of the secondary ray.
    #Normal is into surface by convention (means we get
    #rid of some negatives in the equation...). 
    #Expects normalised vectors!
    mu = n_1/n_2
    parallel = np.sqrt(1 - mu**2 * (1 - (np.dot(incident,normal)**2)))\
             * normal
    perpendicular = mu * (incident - np.dot(incident, normal)*normal )
    return parallel + perpendicular

def refracted_ray(intersection, refracted_dir):
    return Ray(intersection, refracted_dir) 

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