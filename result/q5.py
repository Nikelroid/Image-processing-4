import math
import sys
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def viterbi(contours, count, d, center, a, g, b, mass):
    cost = np.infty
    for j_main in range(-1, 2):
        for i_main in range(-1, 2):
            energy = np.zeros((count, 3, 3)) + np.infty
            memory = np.zeros((count, 3, 3))

            for j_start in range(-1, 2):
                for i_start in range(-1, 2):
                    energy[0, j_start + 1, i_start + 1] = e1((contours[0, 0] + j_start, contours[0, 1] + i_start), g) + \
                                                          e2((contours[0, 0] + j_start, contours[0, 1] + i_start),
                                                             (contours[count - 1, 0] + j_main,
                                                              contours[count - 1, 1] + i_main), d, a) + \
                                                          e3(center, (contours[0, 0] + (j_start * theta),
                                                                      contours[0, 1] + (i_start * theta)), b) + \
                                                          e4(contours,
                                                             (contours[0, 0] + j_start, contours[0, 1] + i_start), mass,
                                                             0)

                    memory[0, j_start + 1, i_start + 1] = ((j_start + 1) * 3 + (i_start + 1))

            for path in range(1, count - 1):
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        for j_last in range(-1, 2):
                            for i_last in range(-1, 2):
                                q = e1((contours[path, 0] + j, contours[path, 1] + i), g) + \
                                    e2((contours[path, 0] + j, contours[path, 1] + i),
                                       (contours[path - 1, 0] + j_last, contours[path - 1, 1] + i_last), d, a) + \
                                    energy[path - 1, j_last + 1, i_last + 1] + \
                                    e3(center, (contours[path, 0] + (j * theta), contours[path, 1] + (i * theta)), b) + \
                                    e4(contours, (contours[path, 0] + j, contours[path, 1] + i), mass, path)

                                if q < energy[path, j + 1, i + 1]:
                                    energy[path, j + 1, i + 1] = q
                                    memory[path, j + 1, i + 1] = ((j_last + 1) * 3 + (i_last + 1))

            for j_end in range(-1, 2):
                for i_end in range(-1, 2):
                    q = e1((contours[count - 1, 0] + j_main, contours[count - 1, 1] + i_main), g) + \
                        e2((contours[count - 1, 0] + j_main, contours[count - 1, 1] + i_main),
                           (contours[count - 2, 0] + j_end, contours[count - 2, 1] + i_end), d, a) + \
                        energy[count - 2, j_end + 1, i_end + 1] + \
                        e3(center,
                           (contours[count - 1, 0] + (j_main * theta), contours[count - 1, 1] + (i_main * theta)), b) + \
                        e4(contours, (contours[count - 1, 0] + j_main, contours[count - 1, 1] + i_main), mass,
                           count - 1)

                    if q < energy[count - 1, j_main + 1, i_main + 1]:
                        energy[count - 1, j_main + 1, i_main + 1] = q
                        memory[count - 1, j_main + 1, i_main + 1] = ((j_end + 1) * 3 + (i_end + 1))

            if cost > energy[count - 1, j_main + 1, i_main + 1]:
                cost = energy[count - 1, j_main + 1, i_main + 1]
                mem = memory
                en = energy
    # print(en)

    t, t, minLoc, t = cv2.minMaxLoc(en[count - 1])
    contours[count - 1] = (minLoc[1] - 1 + contours[count - 1, 0], minLoc[0] - 1 + contours[count - 1, 1])
    pos = (minLoc[1], minLoc[0])
    for path in range(count - 1, 0, -1):
        last = mem[path, pos[0], pos[1]]
        contours[path - 1] = (contours[path - 1, 0] - 1 + int(last / 3), contours[path - 1, 1] - 1 + int(last % 3))
        pos = (int(last / 3), int(last % 3))

    return contours


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def e1(point, g):
    return (-g * (sobelX[point[0], point[1]] ** 2 + sobelY[point[0], point[1]] ** 2))


def e2(point1, point2, d_bar, a):
    return (a * (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 - d_bar)) ** 2)


def e3(point1, point2, b):
    if b == 0:
        return 0
    return b * (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def e4(contours, point, mass, i):
    if mass == 0:
        return 0
    gravity = 0
    count = 0
    i1 = i + 22
    i2 = i + 12
    i3 = i - 12
    i4 = i - 22
    for c in range(len(contours)):
        if i1 > c > i2  or i3 > c > i4:
            gravity += np.abs(c-i) / dist(contours[c], point)
            count += 1
    return -mass * gravity / count


def dist(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def q5(X, Y):
    equality = 0
    a = alpha
    g = gamma
    b = betta
    m = 0
    height, width, layers = org_image.shape
    center = (X[0], Y[0])
    video = cv2.VideoWriter('contour.mp4', -1, 1, (width, height))

    c = contours - 1
    radius = math.sqrt((X[1] - X[0]) ** 2 + (Y[1] - Y[0]) ** 2)
    conts = np.round(np.array(
        [(
         Y[0] + (radius * math.sin(2 * math.pi * i / contours)), X[0] + (radius * math.cos(2 * math.pi * i / contours)))
         for i in
         range(contours)], dtype='int'))

    d_bar = 0
    dd = False
    for steps in range(stp + 1):

        tmp = d_bar
        d_bar = 0
        for i in range(contours - 1):
            d_bar += dist(conts[i + 1], conts[i])
        d_bar += dist(conts[c], conts[0])
        d_bar /= contours * 1.6
        if dd:
            a = 60 - 50 * (steps / stp)
            b = 70 * (steps / stp)**4
            g = 50 + 150 * (steps / stp)**2.7
            m = 300000000*steps
        center = (np.mean(conts[:, 0]), np.mean(conts[:, 1]))
        # print(d_bar)
        temp_image = org_image.copy()

        for i in range(c):
            cv2.line(temp_image, (conts[i, 1], conts[i, 0]), (conts[i + 1, 1], conts[i + 1, 0]), (0, 0, 250), 2)
            cv2.circle(temp_image, (conts[i, 1], conts[i, 0]), 3, (0, 255, 255), 2)
        cv2.line(temp_image, (conts[c, 1], conts[c, 0]), (conts[0, 1], conts[0, 0]), (0, 0, 250), 2)
        cv2.circle(temp_image, (conts[0, 1], conts[0, 0]), 3, (0, 255, 255), 2)
        cv2.circle(temp_image, (conts[c, 1], conts[c, 0]), 3, (0, 255, 255), 2)

        temporary = conts.copy()
        conts = viterbi(conts, contours, d_bar, center, a, g, b, m)
        if np.array_equal(conts, temporary):
            equality += 1
        if equality > 5:
            print('Second part -', steps, '/', stp)
            dd = True
        else:
            print('First part -', steps, '/', stp)
        if equality > 6:
            print('Finish!')
            break
        video.write(temp_image)
        cv2.imshow('Tasbih', temp_image)
        cv2.waitKey(1)


    cv2.imwrite('res11.jpg', temp_image)



def operate(event, x, y, arg1, arg2):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 3, (0, 0, 255), 2)
        Xcoords.append(x)
        Ycoords.append(y)
    # saves clicked points in Xcoords and Ycoords
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(image, (x, y), (Xcoords[0], Ycoords[0]), (0, 255, 0), 2)
        Xcoords.append(x)
        Ycoords.append(y)
        # if the click is not the first one, a line will draw between new click and last one
        if len(Xcoords) == 2:
            r = int(np.round(math.sqrt((Xcoords[1] - Xcoords[0]) ** 2 + (Ycoords[1] - Ycoords[0]) ** 2)))
            cv2.circle(image, (Xcoords[0], Ycoords[0]), r, (0, 0, 255), 2)
            cv2.imshow('Tasbih', image)
            q5(Xcoords, Ycoords)
            sys.exit()
        else:
            mouseX, mouseY = x, y


Xcoords = []
Ycoords = []
contours = 100
alpha = 20
gamma = 20
betta = 4
theta = 80

stp = 300
external_c = 5

# get image  from files
org_image = cv2.imread("tasbih.jpg", 1)

# im = cv2.medianBlur(org_image, ksize=13)

im = cv2.medianBlur(org_image, ksize=13)

sobelX = cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=external_c)
sobelY = cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=external_c)
sobelX = gaussian_filter(sobelX, sigma=2)
sobelY = gaussian_filter(sobelY, sigma=2)


# cv2.imwrite('s1.jpg', sobelX)
# cv2.imwrite('s2.jpg', sobelY)
# copy a version of 0.5 scaled original image to image, for proper show
image = org_image.copy()
# set a window for show image
cv2.namedWindow('Tasbih')
# set callback for created image window, as draw_circle function
cv2.setMouseCallback('Tasbih', operate)
# make a true loop for get update image which is showing in window
while (1):
    cv2.imshow('Tasbih', image)
    k = cv2.waitKey(20) & 0xFF
    # if 'esc' clicked, loop will be break and program closes
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
