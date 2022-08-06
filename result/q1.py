import math
import random
import re

import numpy as np
from matplotlib import pyplot as plt



def get_data(f):
    Data = []
    datasize = int(f.readline())
    for i in range(datasize):
        txt = f.readline()
        x = re.split("\s", txt)
        Data.append((float(x[0]), float(x[1])))
    Data = np.array(Data)
    return np.transpose((np.round(Data * 1000)).astype('int')),datasize


def draw_plot(Data, limit, color='blue', size=1):
    plt.scatter(Data[0], Data[1], s=size, linewidths=1, c=color)
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)


def saveshow_plot(name, ratio):
    plt.xlabel('x *' + ratio)
    plt.ylabel('y *' + ratio)
    plt.savefig(name)
    plt.show()

def k_mean(Data,datasize,k):

    start_points = []
    for i in range(k):
        s = random.randrange(0, len(Data[0]))
        start_points.append((Data[0, s], Data[1, s]))
    start_points = np.array(start_points, dtype=np.float64)

    last1 = 0
    last2 = 0
    while 1:
        Data1 = []
        Data2 = []
        cluster = []
        for i in range(datasize):
            minimum = (7000 * 2) ** 2
            for j in range(k):
                distance = (Data[0, i] - start_points[0, j]) ** 2 + \
                           (Data[1, i] - start_points[1, j]) ** 2
                if distance < minimum:
                    minimum = distance
                    index = i
                    cl_num = j
            cluster.append(cl_num)
            if cl_num == 0:
                Data1.append((Data[0, index], Data[1, index]))
            else:
                Data2.append((Data[0, index], Data[1, index]))

        D1 = np.transpose(np.array(Data1, dtype='int'))
        D2 = np.transpose(np.array(Data2, dtype='int'))

        d11 = np.average(D1[0])
        d12 = np.average(D1[1])
        d21 = np.average(D2[0])
        d22 = np.average(D2[1])
        d = 0
        d += (start_points[0, 0] - d11) ** 2
        d += (start_points[1, 0] - d12) ** 2
        d += (start_points[0, 1] - d21) ** 2
        d += (start_points[1, 1] - d22) ** 2

        if d < 0.001 or last2 == d:
            break
        else:
            start_points[0, 0] = d11
            start_points[1, 0] = d12
            start_points[0, 1] = d21
            start_points[1, 1] = d22
        last2 = last1
        last1 = d
    return cluster,start_points

def q1(f, k):
    Data,datasize = get_data(f)
    draw_plot(Data, 7000)
    saveshow_plot('res01.jpg', 'e3')


    cluster,start_points = k_mean(Data,datasize,k)

    D1 = []
    D2 = []
    for i in range (datasize):
        if cluster[i] == 1:
          D1.append((Data[0,i],Data[1,i]))
        else:
          D2.append((Data[0,i],Data[1,i]))


    D1 = np.transpose(np.array(D1, dtype='int'))
    D2 = np.transpose(np.array(D2, dtype='int'))

    draw_plot(D1, 7000, 'red')
    draw_plot(D2, 7000, 'green')
    saveshow_plot('res02.jpg', 'e3')

    nsystem = []
    for i in range(datasize):
        nsystem.append((math.trunc(Data[0, i]/Data[1, i])*100,
                        np.round(math.sqrt(Data[0, i] ** 2 + Data[1, i] ** 2))))
    nsystem = np.transpose(np.array(nsystem, dtype='int'))
    draw_plot(nsystem, 7000)
    saveshow_plot('r0.jpg', 'e3')

    cluster,start_points = k_mean(nsystem,datasize,k)

    D1 = []
    D2 = []
    for i in range (datasize):
        if cluster[i] == 1:
          D1.append((Data[0,i],Data[1,i]))
        else:
          D2.append((Data[0,i],Data[1,i]))


    D1 = np.transpose(np.array(D1, dtype='int'))
    D2 = np.transpose(np.array(D2, dtype='int'))

    draw_plot(D1, 7000, 'red')
    draw_plot(D2, 7000, 'green')
    #draw_plot(start_points, 7000, 'blue', 15)
    saveshow_plot('res04.jpg', 'e3')



if __name__ == '__main__':
    k = 2
    f = open("Points.txt", "r")
    q1(f, k)
    f.close()
