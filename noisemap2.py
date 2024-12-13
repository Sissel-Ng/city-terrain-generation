from PIL import Image as im
from PIL import ImageShow as ims
from perlin_noise import PerlinNoise
import numpy as np
import time, math
import matplotlib.pyplot as plt

seed = 73
imagesize = 1024

expo = 2.2 #k

expo2 = 5.5 #p

#Value when x=0, (m)
mid = 0.4 

#0.49-0.51 (result), the difference for y-axis (t)
centralrange = 0.004 

#Upper than bound -> 1, lower than -bound -> 0 (u)
bound = 0.9

#-0.02-0.02 (input), the difference for x-axis (h)
central = 0.05 

np.random.seed(seed)

def generate_perlin_noise(shape, res):

    def f(t):

        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])

    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients

    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)

    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)

    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)

    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)

    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps

    n00 = np.sum(grid * g00, 2)

    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)

    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)

    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)

    # Interpolation

    t = f(grid)

    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10

    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11

    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise(shape, res, octaves=1, persistence=1/3):

    noise = np.zeros(shape)

    frequency = 1

    amplitude = 1

    for _ in range(octaves):

        noise += amplitude * generate_perlin_noise(shape, (frequency*res[0], frequency*res[1]))

        frequency *= 2

        amplitude *= persistence

    return noise

def exponfun1(input:float) -> float: #For x<-central

    out = (mid + centralrange) * (1 - ((np.abs(input+central))**(expo)).real/((np.abs(bound - centralrange))**(expo)).real)\
    
    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun1!")

    return 0 if 0 < out < 1e-14 else out

def exponfun2(input:float) -> float: #For |x|<central

    out = np.cos(np.pi / (2*central) * (input + central)) * centralrange + mid

    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun2!")

    return out

def exponfun3(input:float) -> float: #For x>central

    coeff = (1-mid+centralrange)/((bound**expo2).real - (central**expo2).real)

    out = coeff*((input**expo2).real - (central**expo2).real) - centralrange + mid

    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun3!")

    return out

#=================================================

def rounding_by_expo(sam:float) -> float:

    if sam <= -1*bound:

        return 0
            
    if sam <= -1*central:

        return exponfun1(sam)

    if sam <= central:

        return exponfun2(sam)

    if sam <= bound:

        return exponfun3(sam)

    return 1


#=================================================  


noise = generate_fractal_noise((4096, 4096), (8, 8), 8)

plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.colorbar()
plt.show()

noise2 = [[rounding_by_expo(nk) for nk in aj] for aj in noise]

nk = [a for u in noise2 for a in u]
la = len(nk)
print(la)
print(nk[int(la/10*3)])


plt.imshow(noise2, cmap='gray', interpolation='lanczos')
plt.colorbar()
plt.show()