from PIL import Image as im
from PIL import ImageShow as ims
from opensimplex import *
import numpy as np
import time, math

#default seed: 5
#size: 512
#shortenfactor (frequency factor) = 0.00073
#getaway variable: 11.42024
#exponent for smoother lower than midpt: 1.912
#exponent for smoother higher than midpt: 2.936
#the midpoint divider (for 2 different smoother algorithm) = 0.4652
#Copyright 2024 Sissel Ng

seed(75)
imagesize = 2 + 1024
shortfactor = 0.00065 * 4
getaway = 11.42024
fraction = [(0.62,3),(0.94,1),(.27,1/7),(-0.1, 1/67),(-0.32, 0.1576)] #(amp, freq)
fraction2 = [(1/300,105),(1.11,1/36),(.763,1/319)] #Amplitude should be very small!
expo1 = 11.1
expo2 = 3
midpt = 0.47

#function

def fillzero(k:int) -> str: #always in 0-255

    if k >= 100:

        return str(k)
    
    if k >= 10:

        return '0' + str(k)

    return '00' + str(k)

def expofun1(inter:float) -> float:

    return (1-math.pow(1-inter, expo1)) * midpt

def expofun2(inter:float) -> float:

    examp = math.pow(2, (1-expo2)*math.log2(1-midpt))

    return (examp * math.pow((inter - midpt), expo2) + midpt)

def noisefunction(x:int, y:int) -> float:

    noisecom = sums = 0

    for amp, frq in fraction:

        noisecom += amp * noise4( frq * (shortfactor*x + getaway) , frq * (shortfactor*y + getaway), frq/2 * (shortfactor*x + getaway) , frq/2 * (shortfactor*y + getaway) )

        sums += amp

    for amp2, frq2 in fraction2:

        noisecom += amp2 * noise2( frq2 * (shortfactor*x + getaway) , frq2 * (shortfactor*y + getaway))

        sums += amp2

    #noisesmooth = 127.5 * ( 1 + noisecom / sums)
    noiseinterm = 0.5 + noisecom / (sums * 2)
    #noisesmooth = 255 * math.pow(noiseinterm, expo)

    if noiseinterm > midpt:

        noisesmooth = 255 * expofun2(noiseinterm)

    else:

        noisesmooth = 255 * expofun1(noiseinterm) #'''
    
    #the noisecom after the divisor should eventually result in range of -1.0 to 1.0
    #127.5 as 0.0 to 2.0 -> 0.0 to 255.0

    return noisesmooth

def rounding(source:list, x:int, y:int) -> int:

    if (x < 2 or y < 2 or x > imagesize - 3 or y > imagesize - 3):

        return 0

    out = source[x][y]
    + 1/6 * (source[x][y+1] + source[x+1][y] + source[x-1][y] + source[x][y-1])
    + 1/24 * (source[x][y+2] + source[x+2][y] + source[x-2][y] + source[x][y-2])

    return int(24 * out / 29)

def picout(firstarray:list) -> None:

    dou = [[int(firstarray[x][y] if (x < 2 or y < 2 or x > imagesize - 3 or y > imagesize - 3) else rounding(firstarray, x, y) ) for y in range(imagesize)] for x in range(imagesize)]

    with open('sam.txt', 'w') as h:

        for xline in dou:

            h.write('---'.join([fillzero(item) for item in xline]) + '\n')

    noisearray = np.array(dou, dtype=np.uint8)

    with open('sam2.txt', 'w') as h:

        for xline in noisearray:

            h.write('---'.join([fillzero(item) for item in xline]) + '\n')

    out = im.fromarray(noisearray, mode='L')

    ims.show(out)
    out.save('noise.png')


#============================================================================

firstarray = []

print('Start counting...')

st = time.time()

for x in range(imagesize):

    temp = []

    for y in range(imagesize):

        mid = noisefunction(x,y)

        if x > 2 and y > 2:

            pass

        temp.append(mid) 

    firstarray.append(temp)

print(f'Noise finish!, used time (second): {time.time()-st}.')

picout(firstarray)


