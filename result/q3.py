import math
import sys

import cv2
import numpy as np
from scipy import ndimage


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def q3(image, k=64, it=1):
    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).astype('int')
    N = image.shape[0] * image.shape[1]
    S = math.sqrt(N / k)
    h_main = image.shape[0]
    w_main = image.shape[1]
    s = int(np.ceil(S))
    h = int(np.round(image.shape[0] / S))
    w = int(np.round(image.shape[1] / S))
    centers = np.array([[(0, 0) for i in range(w)] for j in range(h)])
    dual = np.array([[0 for i in range(image.shape[1])] for j in range(image.shape[0])])
    Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype('int')
    c_num = 0
    for j in range(h):
        for i in range(w):
            y = int(np.round((S / 2) + (S * j)))
            x = int(np.round((S / 2) + (S * i)))
            t, t, minLoc, t = cv2.minMaxLoc(laplacian[y - 2:y + 3, x - 2:x + 3])
            tuple = (y + minLoc[1], x + minLoc[0])
            centers[j, i] = tuple
            s2 = int(s / 2)
            dual[max(0, tuple[0] - s2):min(h_main, tuple[0] + s2 + 1),
            max(0, tuple[1] - s2):min(w_main, tuple[1] + s2 + 1)] = c_num
            c_num += 1
            # cv2.circle(image, (centers[j, i, 1], centers[j, i, 0]), 1, (0, 0, 255), 1)
            # cv2.circle(laplacian, (centers[j, i, 1], centers[j, i, 0]), 1, 255, 1)
            # print(minLoc[1],minLoc[0])

    # np.set_printoptions(threshold=sys.maxsize)
    # print(s)
    # print(dual)
    temp = np.subtract(np.zeros_like(dual), 1)
    alpha = 6 / S
    inf = (s * 2) + S ** 2 * (256 * 3)
    # print(centers)
    for tu in range(it):
        print('Startint',it,'step')
        for j in range(h + 1):
            for i in range(w + 1):
                coords = np.array([0, 0, 0, 0])

                if i < w and j < h:
                    c0, c1 = centers[j, i, 0], centers[j, i, 1]
                    coords[0] = (dual[c0, c1])
                    l0, a0, b0 = Lab[c0, c1, 0], Lab[c0, c1, 1], Lab[c0, c1, 2]
                    up_y = c0
                    up_x = c1

                if i != 0 and j != h:
                    p1y, p1x = centers[j, i - 1, 0], centers[j, i - 1, 1]
                    coords[1] = (dual[p1y, p1x])
                    l1, a1, b1 = Lab[p1y, p1x, 0], Lab[p1y, p1x, 1], Lab[p1y, p1x, 2]
                    down_x = p1x
                    up_y = p1y

                if j != 0 and i != w:
                    p2y, p2x = centers[j - 1, i, 0], centers[j - 1, i, 1]
                    coords[2] = (dual[p2y, p2x])
                    l2, a2, b2 = Lab[p2y, p2x, 0], Lab[p2y, p2x, 1], Lab[p2y, p2x, 2]
                    down_y = p2y
                    up_x = p2x

                if j != 0 and i != 0:
                    p3y, p3x = centers[j - 1, i - 1, 0], centers[j - 1, i - 1, 1]
                    coords[3] = (dual[p3y, p3x])
                    l3, a3, b3 = Lab[p3y, p3x, 0], Lab[p3y, p3x, 1], Lab[p3y, p3x, 2]
                    down_x = p3x
                    down_y = p3y

                if i == w: up_x = w_main
                if j == h: up_y = h_main
                if i == 0: down_x = 0
                if j == 0: down_y = 0

                for s0 in range(down_y, up_y):
                    for s1 in range(down_x, up_x):

                        d = [inf, inf, inf, inf]

                        if j != h and i != w:
                            d[0] = ((alpha * math.sqrt((s0 - c0) ** 2 + (s1 - c1) ** 2)) \
                                    + math.sqrt((Lab[s0, s1, 0] - l0) ** 2 + (Lab[s0, s1, 1] - a0) ** 2 + (
                                            Lab[s0, s1, 2] - b0) ** 2))

                        if i != 0 and j != h:
                            d[1] = ((alpha * math.sqrt((s0 - p1y) ** 2 + (s1 - p1x) ** 2)) \
                                    + math.sqrt((Lab[s0, s1, 0] - l1) ** 2 + (Lab[s0, s1, 1] - a1) ** 2 + (
                                            Lab[s0, s1, 2] - b1) ** 2))

                        if j != 0 and i != w:
                            d[2] = ((alpha * math.sqrt((s0 - p2y) ** 2 + (s1 - p2x) ** 2)) \
                                    + math.sqrt((Lab[s0, s1, 0] - l2) ** 2 + (Lab[s0, s1, 1] - a2) ** 2 + (
                                            Lab[s0, s1, 2] - b2) ** 2))
                        if j != 0 and i != 0:
                            d[3] = ((alpha * math.sqrt((s0 - p3y) ** 2 + (s1 - p3x) ** 2)) \
                                    + math.sqrt((Lab[s0, s1, 0] - l3) ** 2 + (Lab[s0, s1, 1] - a3) ** 2 + (
                                            Lab[s0, s1, 2] - b3) ** 2))

                        temp[s0, s1] = coords[np.argmin(d)]
                        # print(d)
                        # print(dual[s0, s1])
                print(i, j, '/', w, h)

        dual = ndimage.median_filter(temp, size=int(math.sqrt(S)*2.5))

        c_num = 0
        err_count = 0
        threshold = S / 10
        if tu == it - 1:
            print('FINISH')
            break
        cc = 0
        for j in range(h):
            for i in range(w):
                if c_num in dual:
                    mod = np.zeros_like(dual)
                    mod[dual == c_num] = 1
                    ah = int(np.round(np.sum(np.transpose(mod).dot(range(h_main))) / np.count_nonzero(mod)))
                    aw = int(np.round(np.sum(mod.dot(range(w_main))) / np.count_nonzero(mod)))
                    c0, c1 = centers[j, i, 0], centers[j, i, 1]
                    distance = (ah - c0) ** 2 + (aw - c1) ** 2
                    print(centers[j, i], '->', (ah, aw))
                    centers[j, i] = (ah, aw)
                    if distance < threshold:
                        err_count += 1
                    cc+=1
                c_num += 1
            print(j, '/', h)

        # print("dddddddddddddddd", err_count, c_num)
        print(err_count, '->', cc)
        if err_count < cc * 0.2:
            print('FINISH')
            break
        print(tu + 1, '/ ', it)

    dual = np.array(dual, dtype='uint8')
    dual = cv2.Canny(dual, 0, 0)
    dual = cv2.blur(dual, (3,3))
    dual = np.multiply(np.array(dual, dtype='int'), 255)

    image = np.array(image, dtype='int')
    image = np.add(image, cv2.merge((dual, dual, dual)))
    image[image > 255] = 0
    image = np.array(image, dtype='uint8')

    return image



if __name__ == '__main__':
    k = int(input('please input k: '))
    image = cv2.imread('slic.jpg', 1)
    result = q3(image, k, 5)
    cv2.imwrite('res06.jpg', result)
