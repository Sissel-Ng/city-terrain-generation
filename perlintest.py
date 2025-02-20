import math
from timeit import default_timer as timer
from numba import njit
import numpy as np
from PIL import Image as im
from PIL import ImageShow as ims
p = list()
permutation = [412, 279, 33, 421, 347, 417, 79, 454, 115, 414, 186, 4, 76, 107, 104, 495, 8, 510, 382, 476, 376, 435, 499, 112, 5, 43, 167, 232, 359, 32, 375, 439, 348, 282, 485, 319, 371, 399, 308, 497, 74, 205,
               394, 52, 117, 321, 215, 31, 245, 221, 18, 116, 468, 189, 415, 88, 286, 108, 233, 16, 463, 17, 507, 147, 217, 183, 325, 506, 250, 202, 386, 110, 478, 36, 361, 187, 457, 21, 179, 131, 103, 257, 225, 343, 11,
               388, 151, 318, 340, 255, 59, 95, 369, 327, 94, 67, 197, 39, 322, 364, 352, 14, 141, 368, 304, 344, 440, 475, 169, 50, 464, 503, 100, 72, 58, 292, 73, 306, 283, 393, 328, 37, 425, 378, 354,
               480, 68, 89, 146, 293, 224, 294, 42, 307, 247, 210, 443, 427, 341, 323, 152, 502, 60, 268, 109, 91, 333, 34, 346, 150, 213, 81, 501, 86, 153, 299, 192, 228, 429, 98, 358, 422, 199, 20, 460, 246, 38, 337,
               156, 77, 483, 398, 236, 390, 508, 13, 391, 41, 479, 56, 335, 155, 252, 126, 175, 409, 317, 370, 365, 387, 97, 281, 256, 334, 461, 49, 111, 271, 411, 1, 385, 482, 35, 105, 295, 408, 248, 193, 120, 0,
               312, 453, 474, 272, 494, 188, 481, 184, 331, 329, 357, 227, 403, 137, 241, 185, 441, 446, 360, 296, 338, 208, 436, 355, 206, 214, 62, 264, 209, 90, 162, 46, 332, 470, 211, 161, 125,
               158, 379, 133, 165, 40, 484, 416, 291, 140, 23, 455, 351, 207, 129, 314, 392, 148, 498, 326, 9, 490, 320, 149, 273, 285, 290, 487, 362, 336, 492, 488, 6, 276, 303, 253, 316, 191, 426, 249,
               372, 212, 182, 70, 305, 432, 309, 119, 345, 63, 113, 410, 473, 261, 222, 288, 462, 106, 310, 66, 83, 496, 118, 442, 397, 450, 87, 92, 230, 181, 240, 431, 82, 269, 270, 413, 173, 400, 130, 433, 313,
               493, 356, 178, 136, 505, 44, 204, 377, 216, 235, 339, 300, 61, 451, 251, 259, 406, 226, 96, 127, 452, 2, 22, 263, 350, 302, 349, 407, 367, 260, 219, 401, 287, 380, 489, 458, 405, 297, 437,
               234, 469, 12, 465, 424, 244, 102, 459, 430, 54, 466, 164, 262, 223, 99, 174, 10, 374, 449, 301, 395, 456, 48, 315, 486, 402, 267, 342, 254, 122, 159, 195, 121, 78, 71, 330, 200, 3,
               194, 128, 145, 114, 101, 491, 220, 258, 7, 176, 384, 163, 298, 289, 353, 420, 239, 154, 171, 243, 57, 65, 242, 132, 27, 170, 180, 166, 143, 69, 231, 196, 203, 85, 511, 477, 275, 500, 266, 144, 383, 404,
               160, 124, 467, 277, 419, 19, 284, 363, 381, 445, 218, 123, 201, 229, 418, 396, 190, 29, 64, 134, 237, 53, 423, 177, 55, 24, 324, 238, 30, 25, 80, 28, 389, 274, 15, 444, 93, 75,
               135, 280, 509, 472, 138, 47, 438, 172, 139, 434, 448, 447, 278, 265, 504, 26, 198, 51, 311, 366, 45, 157, 373, 428, 471, 168, 84, 142]
pno = len(permutation)
for i in range(2*pno):
    p.append(permutation[i % pno])
p = np.array(p, int)

@njit
def perlin_noise(x, y, z, permu):
    fade = lambda t : t ** 3 * (t * (t * 6 - 15) + 10)
    lerp = lambda t,a,b : a + t * (b - a)
    X = math.floor(x) & (pno-1)                  # FIND UNIT CUBE THAT
    Y = math.floor(y) & (pno-1)                  # CONTAINS POINT.
    Z = math.floor(z) & (pno-1)
    x -= math.floor(x)                                # FIND RELATIVE X,Y,Z
    y -= math.floor(y)                                # OF POINT IN CUBE.
    z -= math.floor(z)
    u = fade(x)                                # COMPUTE FADE CURVES
    v = fade(y)                                # FOR EACH OF X,Y,Z.
    w = fade(z)
    A = permu[X  ]+Y; AA = permu[A]+Z; AB = permu[A+1]+Z      # HASH COORDINATES OF
    B = permu[X+1]+Y; BA = permu[B]+Z; BB = permu[B+1]+Z      # THE 8 CUBE CORNERS,

    return lerp(w, lerp(v, lerp(u, grad(permu[AA  ], x  , y  , z   ),  # AND ADD
                                   grad(permu[BA  ], x-1, y  , z   )), # BLENDED
                           lerp(u, grad(permu[AB  ], x  , y-1, z   ),  # RESULTS
                                   grad(permu[BB  ], x-1, y-1, z   ))),# FROM  8
                   lerp(v, lerp(u, grad(permu[AA+1], x  , y  , z-1 ),  # CORNERS
                                   grad(permu[BA+1], x-1, y  , z-1 )), # OF CUBE
                           lerp(u, grad(permu[AB+1], x  , y-1, z-1 ),
                                   grad(permu[BB+1], x-1, y-1, z-1 ))))

@njit
def grad(hash, x, y, z):
    h = hash & 15                      # CONVERT LO 4 BITS OF HASH CODE
    u = x if h<8 else y                # INTO 12 GRADIENT DIRECTIONS.
    v = y if h<4 else (x if h in (12, 14) else z)
    return (u if (h&1) == 0 else -u) + (v if (h&2) == 0 else -v)

yaxis = 10240
xaxis = 10240

@njit(parallel=True)
def perlincurve(x,y,offset,oth=0):

  '''out = .62 * perlinout(a,b)
  + .32 * perlinout(a / 9731,b / 19101) #Broader terrain
  + 1/25500 * (perlinout(a*13, b*11))*(perlinout(a*3-1,b*4)) #Small noise
  + .05 * math.sin(perlinout(a*3,b*3+121)) #Medium Noise
, dtype=np.uint8
  if out > 255:

    print(f"{a}-{b} resulted in {out}, more than 1...\n")'''

  inputa = lambda x,y : (x+731)/4352
  inputb = lambda x,y : (y+39)/6172
  inputc = lambda x,y : (x+2*y+689)/8964
  perlinout = lambda t,y : perlin_noise(inputa(t,y), inputb(t,y), inputc(t,y),p)

  out = np.empty((y, x))

  for a in range(x):

    for b in range(y):

      aa = offset + a
      bb = offset + b

      inrim = 3/8 * perlinout(aa,bb)
      + 1/8 * perlinout(2.3*aa,2.3*bb)
      + 1/16 * perlinout(4.32*aa,4.32*bb)
      + 7/16 * perlinout(aa/1000,bb/1000)

      out[a][b] = inrim / 2.08 + .5
  
  out = out ** 2.4 

  out *= 255

  return out

tnow = timer()
nar = perlincurve(xaxis, yaxis, 111.1).astype(np.uint8)
out = im.fromarray(nar, mode='L')
name = f'noise-perlin-{int(timer())}.png'
out.save(name)
print(timer()-tnow)