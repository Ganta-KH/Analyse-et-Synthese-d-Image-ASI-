import numpy as np
from tools import correct_image

def RGB_to_GRAY(image):
    return np.round(np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])).astype(int)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def RGB_to_BIN(image, mn=127, mx=255):
    if len(image.shape) == 3: image = RGB_to_GRAY(image)
    np.putmask(image, image <= mn, 0)
    np.putmask(image, image > mn, mx)
    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def RGB2HSV(image):
    image = np.array(image, dtype='float64')
    HSV = np.zeros_like(image, dtype='float64')
    R, G, B = image[..., 0] / 255., image[..., 1] / 255., image[..., 2] / 255.
    Cmax = np.max(image[..., :3], axis=-1)
    Cmin = np.min(image[..., :3], axis=-1)
    delta = Cmax - Cmin
    i, j = np.where(delta == 0)
    HSV[i, j, 2] = 0
    HSV[:, :, 2] += np.multiply((Cmax == R), (60 * ((np.divide((G - B), delta)) % 6) ))
    HSV[:, :, 2] += np.multiply((Cmax == G), (60 * ((np.divide((B - R), delta)) + 2) ))
    HSV[:, :, 2] += np.multiply((Cmax == B), (60 * ((np.divide((R - G), delta)) + 4) ))

    i, j = np.where(Cmax == 0)
    HSV[i, j, 1] = 0
    HSV[:, :, 1] = np.divide(delta, Cmax)

    HSV[:, :, 0] = Cmax
    HSV[:, :, 2] *= 255.
    HSV[:, :, 1] *= 255.
    HSV[:, :, 0] /= 2.
    return HSV

def RGB_to_HSV(image):
    newImg = []
    for i in range( len(image) ):
        newImg.append([])
        for j in range( len(image[0]) ):
            RGB = [image[i][j][0] / 255.0, image[i][j][1] / 255.0, image[i][j][2] /255.0]
            Cmax = max(RGB)
            Cmin = min(RGB)
            delta = Cmax - Cmin
            # HUE
            if delta == 0: H = 0
            elif Cmax == RGB[0]: H = 60 * (((RGB[1] - RGB[2]) / delta) % 6)
            elif Cmax == RGB[1]: H = 60 * (((RGB[2] - RGB[0]) / delta) + 2)
            elif Cmax == RGB[2]: H = 60 * (((RGB[0] - RGB[1]) / delta) + 4)
            # Saturation
            if Cmax == 0: S = 0
            else: S = delta / Cmax
            # Value
            V = Cmax
            newImg[i].append([V * 255, S*255, H/2])
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def HSV_to_RGB(image):
    newImg = []
    RGB = [0, 0, 0]
    for i in range( len(image)):
        newImg.append([])
        for j in range( len(image[0])):
            HSV = [image[i][j][2] * 2, image[i][j][1] / 255, image[i][j][0] / 255]
            C = HSV[2] * HSV[1]
            X = C * (1 - abs( (HSV[0] / 60) % 2 - 1 ))
            m = HSV[2] - C
            if 0 <= HSV[0] < 60: RGB = [C, X, 0]
            elif 60 <= HSV[0] < 120: RGB = [X, C, 0]
            elif 120 <= HSV[0] < 180: RGB = [0, C, X]
            elif 180 <= HSV[0] < 240: RGB = [0, X, C]
            elif 240 <= HSV[0] < 300: RGB = [X, 0, C]
            elif 300 <= HSV[0] < 360: RGB = [C, 0, X]

            newImg[i].append( [(RGB[0] + m) * 255, (RGB[1] + m) * 255, (RGB[2] + m) * 255] )
    newImg = correct_image(np.array(newImg))
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def RGB_to_YCrCb(image):
    Y = RGB_to_GRAY(image)
    Cr = (image[:, :, 0] - Y) * 0.713 + 128
    Cb = (image[:, :, 2] - Y) * 0.564 + 128
    image[:, :, 0] = Cb
    image[:, :, 1] = Cr
    image[:, :, 2] = Y
    return image
"""
def RGB_to_YCrCb(image):
    newImg = []
    for i in range(len(image)):
        newImg.append([])
        for j in range(len(image[0])):
            Y = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
            Cr = (image[i][j][0] - Y) * 0.713 + 128
            Cb = (image[i][j][2] - Y) * 0.564 + 128
            newImg[i].append([Cb, Cr, Y])
    return newImg
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
def YCrCb_to_RGB(image):
    R = image[:, :, 2] + 1.403 * (image[:, :, 1] - 128)
    G = image[:, :, 2] + 0.714 * (image[:, :, 1] - 128) - 0.344 * (image[:, :, 0] - 128)
    B = image[:, :, 2] + 1.773 * (image[:, :, 0] - 128)
    image[..., :, 0] = R
    image[..., :, 1] = G
    image[..., :, 2] = B
    np.putmask(image, image > 255, 255)
    np.putmask(image, image < 0, 0)
    return image
"""
def YCrCb_to_RGB(image):
    newImg = []
    for i in range(len(image)):
        newImg.append([])
        for j in range(len(image[0])):
            R = image[i][j][2] + 1.403 * (image[i][j][1] - 128)
            G = image[i][j][2] - 0.714 * (image[i][j][1] - 128) - 0.344 * (image[i][j][0] - 128)
            B = image[i][j][2] + 1.773 * (image[i][j][0] - 128)
            newImg[i].append([R, G, B])
    newImg = correct_image(np.array(newImg))
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
