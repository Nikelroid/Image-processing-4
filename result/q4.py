import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def q4(mask):
    print('working')

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.array(mask[:int(mask.shape[0] / 4), :int(mask.shape[1] / 4)], dtype='int')
    mask[(mask > 0)] = 1
    mask[mask == -1] = 2
    mask = np.array(mask,dtype='uint8')
    im = resize(org_image,0.25)
    mask, bgdModel, fgdModel = cv2.grabCut(im, mask, None, bgdModel, fgdModel, 35, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask_main = cv2.resize(mask, (0, 0), fx=4, fy=4)

    print('done')
    for m in range (mask_main.shape[0]):
        if np.max(mask_main[m])==0:
            continue
        for n in range(mask_main.shape[1]):
            if np.max(mask_main[m,n]) != 0:
                org_image[m,n] = (int(org_image[m,n,0]/1.5),128,
                                  int(org_image[m,n,2]/1.5))

    cv2.imwrite('res10.jpg', org_image)



def operate(event, x, y, arg1, arg2):
    global mask_label
    global mouseX, mouseY

    if event == cv2.EVENT_LBUTTONDOWN and mask_label==1:
        try:
            image[y - r:y + r + 1, x - r:x + r + 1] = radial31
            mask[y - r:y + r + 1, x - r:x + r + 1] = \
                np.add(mask[y - r:y + r + 1, x - r:x + r + 1], radial2)
        except:
            print('Out of range')


    if event == cv2.EVENT_RBUTTONDOWN:
        if mask_label == 0:
            mask_label = 1
        elif mask_label == 1:
            mask_label = 0

    if event == cv2.EVENT_MOUSEMOVE:
        if mask_label == 0:
            try:
                image[y - r0:y + r0 + 1, x - r0:x + r0 + 1] = radial30
                image[image < 0] = 0
                mask[y - r0:y + r0 + 1, x - r0:x + r0 + 1] = \
                    np.multiply(mask[y - r0:y + r0 + 1, x - r0:x + r0 + 1], radial4)
            except:
                print('Out of range')


    image[image > 255] = 255
    cv2.imshow('Birds', image)
    cv2.waitKey(1)
    # saves clicked points in Xcoords and Ycoords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        q4(mask)
        sys.exit()
    mouseX, mouseY = x, y


Xcoords = []
Ycoords = []
mask_label = 1
r = 3
r0 = 7
radial30 = np.full((2 * r0 + 1, 2 * r0 + 1, 3), (0, 0, 255))
radial31 = np.full((2 * r + 1, 2 * r + 1, 3), (255, 0, 0))

radial4 = np.full((2 * r0 + 1, 2 * r0 + 1), 0)
radial2 = np.full((2 * r + 1, 2 * r + 1), 2)



# get image  from files
org_image = cv2.imread("birds.jpg", 1)


mask = np.zeros(org_image.shape[:2], dtype='int') - 1
main_image = resize(org_image.copy(), 0.5)
image = resize(org_image.copy(), 0.25)
# set a window for show image
cv2.namedWindow('Birds')

cv2.setMouseCallback('Birds', operate)
# make a true loop for get update image which is showing in window


while (1):
    cv2.imshow('Birds', image)
    k = cv2.waitKey(20) & 0xFF
    # if 'esc' clicked, loop will be break and program closes
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
