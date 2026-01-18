#%%
# ffmpeg   -framerate 25   -pattern_type glob   -i '*.png' -filter_complex "[0]reverse[r];[0][r]concat=n=2:v=1:a=0,split[a][b];[a] palettegen [p];[b][p] paletteuse"  out.gif
# ffmpeg  -i out.gif -vf "fps=10,minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" out_int.gif

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import shutil
import os
from pathlib import Path

from itertools import cycle


PATH = 'brot/'
IMAGE_PATH = 'brot_img/'
for path in [PATH, IMAGE_PATH]:
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

cmap = sns.color_palette('magma', as_cmap=True)

SHOW_IMAGE = False

DTYPE = np.float16

HEIGHT = 9
WIDTH = 16
RATIO = WIDTH/HEIGHT

RES_SPACE = 12
RES_TIME = 99
MAX = 1.5
MIN = -MAX
N_ITER = 17

x = np.linspace(MIN*RATIO, MAX*RATIO, WIDTH*RES_SPACE, dtype=DTYPE)
y = np.linspace(MIN, MAX, HEIGHT*RES_SPACE, dtype=DTYPE)
COMPLEX_PLANE = x + 1j * y[:,None]

MASK = np.ones_like(COMPLEX_PLANE, dtype=bool)

def iterate(time_fast, time_slow=3, ceiling=4, C=COMPLEX_PLANE, M=MASK, N=N_ITER):
    OUT = np.zeros_like(M, dtype=np.uint8)
    Z = np.zeros_like(C)
    C = np.copy(C)
    M = np.copy(M)
    ceiling = ceiling + ceiling*1j
    for n in range(N):
        # M[Z.imag**2 + Z.real**2 > max] = False
        M[Z > ceiling] = False
        Z *= np.exp(-time_fast*10j)
        C *= np.exp(time_fast*1j)
        Z[M] = Z[M]**1.5 + C[M]**-time_slow
        Z[M] *= np.exp(time_fast*C[M]**-3)
        OUT -= M
    OUT *= 15
    return OUT

def shadow(im, sigma=10):
    im_ = gaussian_filter(im, sigma=sigma)
    return np.maximum(im, im_)

if SHOW_IMAGE:
    cv.namedWindow("im", cv.WINDOW_NORMAL)
    cv.setWindowProperty("im", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)


loop_number = 0
J = [3, 2, 1, 1/2, 1/3, 0, -1/2, -1/3, -1, -2, -3]
J =  J[loop_number:]
for j in J:
    frame_number = 0
    I = np.linspace(-np.pi,np.pi, RES_TIME)
    I = I[frame_number:]
    # padding = np.ones((RES_TIME//30))*I[-1]
    # I = np.concatenate((padding, I, padding))
    for i in I:
        try:
            im = np.load(PATH + f'{loop_number:02d}_{frame_number:04d}.npy')
            # im = np.load('')
            print(f'{loop_number:02d}_{frame_number} Already exists.', end='\r')
        except FileNotFoundError:
            zoom = - ((-i + np.pi) / 4)
            C = COMPLEX_PLANE * (np.tan(zoom))

            CEILING = np.exp(np.tan(-i/2)*5)

            im = iterate(i, time_slow=j, ceiling=CEILING, C=C)
            im = (im / im.max()).astype(np.float32)
            np.save(PATH + f'{loop_number:02d}_{frame_number:04d}', im)


        im = shadow(im)
        im = cmap(im, bytes=True)[:,:,:-1]
        plt.imsave(IMAGE_PATH + f'{loop_number:02d}_{frame_number:04d}.png', im)
        if SHOW_IMAGE:
            cv.imshow('im',cv.cvtColor(im, cv.COLOR_RGB2BGR))
            k = cv.waitKey(1)
        frame_number+=1

    for path in [PATH, IMAGE_PATH]:
        zip_path = f'{loop_number:02d}'+path[:-1]
        shutil.make_archive(zip_path, 'zip', path)

    for path in [PATH, IMAGE_PATH]:
        shutil.rmtree(path)
        os.mkdir(path)
    loop_number += 1
    break
# %%
