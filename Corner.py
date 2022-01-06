import numpy as np
import cv2
from EdgeDetection import *


def Harris(image, k, U):
    W, H = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    Ix = convolution(gray, filters('sobel_x'), 1)
    Iy = convolution(gray, filters('sobel_y'), 1)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
    harris_response = detA - k * traceA ** 2
    offset = U//2

    for y in range(offset, W-offset):
        for x in range(offset, H-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    r = det - k*(trace**2)

    img_copy_for_corners = np.copy(image)
    img_copy_for_edges = np.copy(image)

    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > 0:
                # this is a corner
                img_copy_for_corners[rowindex, colindex] = [255,0,0]
            elif r < 0:
                # this is an edge
                img_copy_for_edges[rowindex, colindex] = [0,255,0]

    #return img_copy_for_corners, img_copy_for_edges


    
    corners, edges = img_copy_for_corners, img_copy_for_edges

    cv2.imshow('c', corners)
    cv2.imshow('e', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('Images/sudoku.png')
Harris(img, .5, 5)






