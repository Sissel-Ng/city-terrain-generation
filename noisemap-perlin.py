from perlin_noise import PerlinNoise
from PIL import Image as im
from PIL import ImageShow as ims
import numpy as np
import time, math

#default seed: 17
#size: 512
#shortenfactor (frequency factor) = 0.0032
#getaway variable: 11.42024

seed = 17
imagesize = 1024
shortfactor = 0.00032
getaway = 11.42024
fraction = [(1.8,1,shortfactor),(0.4,2,shortfactor),(.0192,50,shortfactor)]
fraction2 = [(4.1,1/270,shortfactor/2),(1.2,1/51,shortfactor/2)]
expo = 3.3
octave = 4
noise = PerlinNoise(octaves = octave, seed = seed)

#function

def fillzero(k:int) -> str: #always in 0-255

    if k >= 100:

        return str(k)
    
    if k >= 10:

        return '0' + str(k)

    return '00' + str(k)

def noisefunction(x:int, y:int, fraction:list, expo:float) -> float:

    noisecom = sums = 0

    for amp, frq, sf in fraction:

        noisecom += amp * noise( [frq * (sf*x + getaway) , frq * (sf*y + getaway)] )

        sums += amp

    #noisesmooth = 255 * (noisecom / sums) 
    noisesmooth = 255 * math.pow((0.5 + noisecom / (sums * 2)), expo)
    
    #the noisecom after the divisor should eventually result in range of 0 to 1.0
    #127.5 as 0.0 to 2.0 -> 0.0 to 255.0

    return noisesmooth

def rounding(source:list, x:int, y:int) -> int:

    if (x < 2 or y < 2 or x > imagesize - 3 or y > imagesize - 3):

        return 0

    out = source[x][y]
    + 1/6 * (source[x][y+1] + source[x+1][y] + source[x-1][y] + source[x][y-1])
    + 1/24 * (source[x][y+2] + source[x+2][y] + source[x-2][y] + source[x][y-2])

    return int(24 * out / 29)

#===============================================

firstarray = []

print('Start counting...')

st = time.time()

for x in range(imagesize):

    temp = []

    for y in range(imagesize):

        mid = noisefunction(x,y,fraction,expo)

        mid = (2*mid + noisefunction(x+mid, y+mid, fraction2, expo*5)) / 3

        temp.append(mid) 

    firstarray.append(temp)

print(f'Noise finish!, used time (second): {time.time()-st}.')

dou = [[int(firstarray[x][y] if (x < 2 or y < 2 or x > imagesize - 3 or y > imagesize - 3) else rounding(firstarray, x, y) ) for y in range(imagesize)] for x in range(imagesize)]

with open('sam-perlin.txt', 'w') as h:

    for xline in dou:

        h.write('---'.join([fillzero(item) for item in xline]) + '\n')

noisearray = np.array(dou, dtype=np.uint8)

with open('sam2-perlin.txt', 'w') as h:

    for xline in noisearray:

        h.write('---'.join([fillzero(item) for item in xline]) + '\n')

out = im.fromarray(noisearray, mode='L')

ims.show(out)
out.save('noise-perlin.png')
