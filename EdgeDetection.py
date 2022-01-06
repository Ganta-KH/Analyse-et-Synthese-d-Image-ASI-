import numpy as np
from ConversionImage import RGB_to_BIN, RGB_to_GRAY
from tools import correct_image
import cv2
from Threshold import *


def Naif_Detector(image):
    img = RGB_to_BIN(np.array(image))
    verticale = abs(img[:, 0: -1] - img[:, 1:])
    horizantal = abs(img[0: -1, :] - img[1:, :])
    contour = verticale[:-1,:] + horizantal[:, :-1]
    np.putmask(contour, contour > 255, 255)
    return contour

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convolution_calc(image, filter, k):
    W, H = image.shape[:2]
    img = np.zeros((W+k*2, H+k*2))
    img[1*k:-1*k, 1*k:-1*k] = image
    G = np.zeros((W, H))
    for i in range(k, W+k):
        for j in range(k, H+k):
            G[i-k, j-k] = np.sum(np.multiply( img[i-k : i+k+1, j-k : j+k+1], filter ))
    return G

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convolution_calc_3D(image, filter, k):
    W, H = image.shape[:2]
    img = np.zeros((W+k*2, H+k*2, 3))
    img[1*k:-1*k, 1*k:-1*k, :] = image
    RES = np.zeros((W, H, 3))
    R, G, B = np.zeros((W, H)), np.zeros((W, H)), np.zeros((W, H))
    for i in range(k, W+k):
        for j in range(k, H+k):
            R[i-k, j-k] = np.sum(np.multiply( img[i-k : i+k+1, j-k : j+k+1, 0], filter ))
            G[i-k, j-k] = np.sum(np.multiply( img[i-k : i+k+1, j-k : j+k+1, 1], filter ))
            B[i-k, j-k] = np.sum(np.multiply( img[i-k : i+k+1, j-k : j+k+1, 2], filter ))
    RES[:, :, 0] = R
    RES[:, :, 1] = G
    RES[:, :, 2] = B
    return RES

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convolution(image, filter, k):
    if len(image.shape) == 3: return correct_image(convolution_calc_3D(image, filter, k))
    else: return correct_image(convolution_calc(image, filter, k))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def filters(value):
    if value == 'naif_x': return np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    elif value == 'naif_y': return np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    elif value == 'sobel_x': return np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) # sobel_x
    elif value == 'sobel_y': return np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    elif value == 'sobel_z': return np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]) 
    elif value == 'sobel_k': return np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    elif value == 'sobel_o': return np.array([[0, 1, 2],[-1, 0, 1],[2, -1, 0]])
    elif value == 'prewitt_x': return np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]]) / 3
    elif value == 'prewitt_y': return np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]]) / 3
    elif value == 'prewitt_dx': return np.array([[1, 1, 0],[1, 0, -1],[0, -1, -1]]) / 3
    elif value == 'prewitt_dy': return np.array([[0, 1, 1],[-1, 0, 1],[-1, -1, 0]]) / 3
    elif value == 'moyen': return np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) / 9
    elif value == 'gaussian_4': return np.array([[0, 1, 0],[1, 4, 1],[0, 1, 0]]) / 8
    elif value == 'gaussian_8': return np.array([[1, 1, 1],[1, 8, 1],[1, 1, 1]]) / 16
    elif value == 'laplacian_4': return np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    elif value == 'laplacian_8': return np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
    elif value == 'gaussian_3x3': return np.array(([1 ,2 , 1], [2, 4, 2], [1 ,2 ,1])) /16   #1/16 Gaussian blur 3 x 3
    elif value == 'gaussian_5x5': return np.array(([1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1])) /256 # 1/256 Gaussian blur 5 × 5

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Sobel(image):
    Gx = convolution(image, filters('sobel_x'), 1)
    Gy = convolution(image, filters('sobel_y'), 1)  
    return correct_image(np.sqrt(Gx ** 2 + Gy ** 2))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Roberts(image):
    binary = RGB_to_BIN(image)
    Gx = binary[:-1, :-1] - binary[1:, 1:]
    Gy = binary[1:, :-1] - binary[:-1, 1:]
    return np.sqrt(Gx ** 2 + Gy ** 2)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Prewitt(image):
    Gx = convolution(image, filters('prewitt_x'), 1)
    Gy = convolution(image, filters('prewitt_y'), 1)
    #Gz = convolution(image, filters('prewitt_dx'), 1)
    #Gk = convolution(image, filters('prewitt_dy'), 1)
    return np.sqrt(Gx ** 2 + Gy ** 2) #np.sqrt(Gx ** 2 + Gy ** 2 + Gz ** 2 + Gk ** 2)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LoG(image):
    image = convolution(image, filters('gaussian_3x3'), 1)
    return convolution(image, filters('laplacian_4'), 1)

def LoG_CV(image):
    image = cv2.GaussianBlur(image, (3,3), 0)
    return  cv2.Laplacian(image, cv2.CV_64F) #cv2.filter2D(image, -1, filters('laplacian_8'))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def DoG(image):
    gaus3 = convolution(image, filters('gaussian_3x3'), 1)
    gaus5 = convolution(image, filters('gaussian_5x5'), 2)
    newImg = gaus3 - gaus5
    return correct_image(newImg * 255) 

def DoG_CV(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaus3 = cv2.GaussianBlur(image, (3,3), 0)
    gaus5 = cv2.GaussianBlur(image, (5,5), 0)
    return correct_image(cv2.subtract(gaus3, gaus5) * 255)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Non_Maximum_Suppression(image, angle):
    W, H = image.shape
    newImg = np.zeros((W, H))
    angle[angle < 0] += 180

    pixel_1, pixel_2 = 0, 0

    for i in range(1, W-1):
        for j in range(1, H-1):
            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                pixel_1 = image[i, j - 1]
                pixel_2 = image[i, j + 1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                pixel_1 = image[i - 1, j + 1]
                pixel_2 = image[i + 1, j - 1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                pixel_1 = image[i + 1, j]
                pixel_2 = image[i - 1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                pixel_1 = image[i + 1, j - 1]
                pixel_2 = image[i - 1, j + 1]

            if (image[i, j] >= pixel_1) and (image[i, j] >= pixel_2):
                newImg[i, j] = image[i, j]
            else: newImg[i, j] = 0
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Canny(image, mn=100, mx=200):
    gray = RGB_to_GRAY(image)
    blur = convolution(gray, filters('gaussian_5x5'), 2)
    Gx = convolution(blur, filters('sobel_x'), 1)
    Gy = convolution(blur, filters('sobel_y'), 1)
    Gz = convolution(blur, -filters('sobel_x'), 1)
    Gw = convolution(blur, -filters('sobel_y'), 1)
    G, angle = cv2.cartToPolar(Gx, Gy, angleInDegrees = True)
    NMS = Non_Maximum_Suppression(G, angle)
    thresh = Thresholding(NMS, mn ,mx)
    result = Hysteresis(thresh, mn, mx)
    return result

def Canny_CV(image, mn=100, mx=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, mn, mx)
    return canny




