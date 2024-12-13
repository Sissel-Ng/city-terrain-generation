from PIL import Image as im
from PIL import ImageShow as ims
from opensimplex import *
import numpy as np
import time, math

#default seed: 123
#size: 512
#shortenfactor (frequency factor) = 0.00073
#getaway variable: 11.42024
#Copyright 2024 Sissel Ng

seed(123)
imagesize = 2 + 1024*2
shortfactor = .000073
getaway = 11.42024
fraction = [(1,1), (1/5,1/67), (-.3,7), (1/3,5)] #(amp, freq)

arthirounding = [1, 1/21]

expo = 2.2 #k

expo2 = 5.5 #p

#Value when x=0, (m)
mids = 0.4 

#0.49-0.51 (result), the difference for y-axis (t)
centralrange = 0.004 

#Upper than bound -> 1, lower than -bound -> 0 (u)
bounded = 0.9

#-0.02-0.02 (input), the difference for x-axis (h)
central = 0.05 

#function

def fillzero(k:int) -> str: #always in 0-255

    if k >= 100:

        return str(k)
    
    if k >= 10:

        return '0' + str(k)

    return '00' + str(k)

def exponfun1(input:float) -> float: #For x<-central

    out = (mids + centralrange) * (1 - ((np.abs(input+central))**(expo)).real/((np.abs(bounded - centralrange))**(expo)).real)
    
    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun1!")

    return 0 if 0 < out < 1e-14 else out

def exponfun2(input:float) -> float: #For |x|<central

    out = np.cos(np.pi / (2*central) * (input + central)) * centralrange + mids

    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun2!")

    return out

def exponfun3(input:float) -> float: #For x>central

    coeff = (1-mids+centralrange)/((bounded**expo2).real - (central**expo2).real)

    out = coeff*((input**expo2).real - (central**expo2).real) - centralrange + mids

    if out < 0:
    
        raise Exception(f"ValueError! {input} outputing {out} which is lower than 0. Fun3!")

    return out

def rounding_by_expo(sam:float) -> float:

    if sam <= -1*bounded:

        return 0
            
    if sam <= -1*central:

        return exponfun1(sam)

    if sam <= central:

        return exponfun2(sam)

    if sam <= bounded:

        return exponfun3(sam)

    return 1

def noisefunction(x:int, y:int) -> float:

    noisecom = sums = 0

    for amp, frq in fraction:

        noisecom += amp * noise2( frq * (shortfactor*x + getaway) , frq * (shortfactor*y + getaway))

        sums += amp

    noiseinterm = noisecom / sums
    #noisesmooth = 255 * math.pow(noiseinterm, expo)

    noisesmooth = rounding_by_expo(noiseinterm)
    
    #the noisecom after the divisor should eventually result in range of -1.0 to 1.0
    #127.5 as 0.0 to 2.0 -> 0.0 to 255.0

    return noisesmooth

def rounding(source:list, x:int, y:int) -> int:

    end = len(arthirounding)

    out = 0

    vari = [[(0,r),(r,0),(0,-r),(-r,0)] for r in range(1, end)]

    if 0 <= x < end - 1:

        for pk in range(x, end - 1):

            vari[pk][3] = (0,0)
            
    if 0 <= y < end - 1:

        for pk in range(y, end - 1):

            vari[pk][2] = (0,0)

    if imagesize - end - 1 < x < imagesize:

        for jk in range(imagesize - x - 1, end - 1):

            vari[jk][1] = (0,0)

    if imagesize - end - 1 < y < imagesize:

        for jk in range(imagesize - y - 1, end - 1):

            vari[jk][0] = (0,0)

    vari = [[(0,0),(0,0),(0,0),(0,0)]] + vari 

    for ind in range(end):
            
        for u in vari[ind]:

            out += arthirounding[ind] * (source[x+u[0]][y+u[1]])

    return int(out * 255 / (4*sum(arthirounding)))

def picout(firstarray:list) -> None:

    dou = [ [rounding(firstarray, x, y) for y in range(imagesize)] for x in range(imagesize) ]

    with open('sam.txt', 'w') as h:

        for xline in dou:

            h.write('---'.join([fillzero(item) for item in xline]) + '\n')

    noisearray = np.array(dou, dtype=np.uint8)

    out = im.fromarray(noisearray, mode='L')

    ims.show(out)

    out.save('noise.png')

#============================================================================

if __name__ == "__main__":

    firstarray = []

    print('Start counting...')

    st = time.time()

    for x in range(imagesize):

        temp = []

        for y in range(imagesize):

            mid = noisefunction(x,y)

            temp.append(mid) 

        firstarray.append(temp)

    print(f'Noise finish!, used time (second): {time.time()-st}.')

    picout(firstarray)


