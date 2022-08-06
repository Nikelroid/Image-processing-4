import math
import random
import re

import cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def clipping(img):
    # make negative values of img 0
    # make larger than 255 values 255
    return (img.clip(0, 255)).astype('uint8')


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def draw_plot(ax, Data, size=1):
    ax.scatter(Data[0], Data[1], Data[2], c=Data[2], cmap='viridis', linewidth=size)


def saveshow_plot(name):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(name)
    plt.show()


def clip(number, r):
    if number < r:
        return r
    if number > 255 - r:
        return 255 - r
    return number


def q2(image,r):
    histogram = np.array([[[0 for L in range(256)] for u in range(256)] for v in range(256)])
    resolution = image.shape
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            histogram[image[i, j, 0], image[i, j, 1], image[i, j, 2]] += 1
        print(i, '/', resolution[0])
    n = 0
    ################################
    radial = np.zeros((2 * r - 1, 2 * r - 1, 2 * r - 1), dtype=np.float64)
    for i in range(2 * r - 1):
        for j in range(2 * r - 1):
            for k in range(2 * r - 1):
                if (i - r + 1) ** 2 + (j - r + 1) ** 2 + (k - r + 1) ** 2 <= (r - 1) ** 2:
                    radial[i, j, k] = 1
    ################################
    dual = histogram.copy()

    for i in range(256):
        for j in range(256):
            for k in range(256):
                if i < r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[r, j, k] += t
                elif i > 256 - r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[255 - r, j, k] += t
                if j < r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[i, r, k] += t
                elif j > 256 - r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[i, 255 - r, k] += t
                if k < r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[i, j, r] += t
                elif k > 256 - r:
                    t = histogram[i, j, k]
                    histogram[i, j, k] = 0
                    histogram[i, j, 255 - r] += t
        print(i, '/', 256)


    for i in range( r, 256 -  r):
        if np.max(dual[i, :, :]) <= 0:
            continue
        for j in range( r, 256 -  r):
            if np.max(dual[i, j, :]) <= 0:
                continue
            for k in range( r, 256 -  r):
                if np.max(dual[i, j, k]) <= 0:
                    continue
                if dual[i, j, k] < 0:
                    continue
                tmp = []
                px = i
                py = j
                pz = k
                while 1:
                    tmp.append((px, py, pz))
                    found = False
                    matrix = np.multiply(histogram[px - r + 1:px + r, py - r + 1:py + r, pz - r + 1:pz + r], r)
                    dual_matrix = np.multiply(dual[px -  r + 1:px +  r,
                                              py -  r + 1:py +  r,
                                              pz -  r + 1:pz +  r],  radial)
                    s = np.sum(matrix)

                    av_x = 0
                    for x in range(2 * r - 1):
                        av_x += x * np.sum(matrix[x, :, :])
                    av_x = int(np.round((av_x / s + px) - (r - 1)))

                    av_y = 0
                    for y in range(2 * r - 1):
                        av_y += y * np.sum(matrix[:, y, :])
                    av_y = int(np.round((av_y / s + py) - (r - 1)))

                    av_z = 0
                    for z in range(2 * r - 1):
                        av_z += z * np.sum(matrix[:, :, z])
                    av_z = int(np.round((av_z / s + pz) - (r - 1)))
                    rr = 255 - r
                    if not (r < av_x < rr and r < av_y < rr and r < av_z < rr):
                        if np.min(dual_matrix) < 0:
                            mat = np.min(dual_matrix)
                            for tp in tmp:
                                dual[tp] = mat
                        else:
                            n -= 1
                            for tp in tmp:
                                dual[tp] = n
                        found = True
                    if found: break

                    if av_x == px and av_y == py and av_z == pz:
                        if np.min(dual_matrix) < 0:
                            mm = np.min(dual_matrix)
                            for tp in tmp:
                                dual[tp] = mm
                        else:
                            n -= 1
                            for tp in tmp:
                                dual[tp] = n
                        found = True
                    else:

                        if dual[av_x, av_y, av_z] < 0:
                            cl_num = np.min(dual[av_x, av_y, av_z])
                            for tp in tmp:
                                dual[tp] = cl_num
                            found = True
                        if found: break

                        px = av_x
                        py = av_y
                        pz = av_z
                    if found: break
        print(i, '/', 256 - r, '-', -n)

    print(np.max(dual))

    print('Cluster:', -n)
    clusters = [[] for cl in range(-n)]
    wights = [0 for cl in range(-n)]
    for i in range(256):
        if np.min(dual[i, :, :]) >= 0:
            continue
        for j in range(256):
            if np.min(dual[i, j, :]) >= 0:
                continue
            for k in range(256):
                if np.min(dual[i, j, k]) >= 0:
                    continue
                value = histogram[i, j, k]
                array = [i * value, j * value, k * value]
                clusters[-dual[i, j, k] - 1].append(array)
                wights[-dual[i, j, k] - 1] += value
        print(i, '/ 255')
    print('Cluster:', len(clusters))

    cl_averages = [tuple() for x in range(len(clusters))]
    for clstr in range(len(clusters)):
        if not clusters[clstr]:
            cl_averages[clstr] = (0, 0, 0)
        else:
            average = (0, 0, 0)
            for cl in clusters[clstr]:
                average = np.add(average, tuple(cl))
            w = wights[clstr]
            average = (int(np.round(np.divide(average[0], w))), int(np.round(np.divide(average[1], w))),
                       int(np.round(np.divide(average[2], w))))
            cl_averages[clstr] = average
        print(clstr, '/', len(clusters))
    print('Done')

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            d = dual[image[i, j, 0], image[i, j, 1], image[i, j, 2]]
            if d < 0:
                image[i, j] = cl_averages[-d - 1]
        print(i, '/', resolution[0])
    return image


if __name__ == '__main__':
    image = cv2.imread('park.jpg', 1)
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    result = cv2.cvtColor(q2(imageYCrCb,7),cv2.COLOR_YCrCb2BGR)
    result = cv2. medianBlur(result,15)
    cv2.imwrite('res05.jpg', result)
