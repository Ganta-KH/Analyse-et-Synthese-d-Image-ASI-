import cv2
from ConversionImage import *
from Histogram import *
from GeometricTransformations import *
from EdgeDetection import *
from Threshold import *
from Lines_Circels import *

def ImageConversion(image, imageCV, value):
    if value == 1: return RGB_to_BIN(np.array(image)), 'Bin', cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
    elif value == 2: return RGB_to_GRAY(np.array(image)), 'Grey', cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY)
    elif value == 3: return RGB_to_HSV(image), 'HSV', cv2.cvtColor(imageCV, cv2.COLOR_BGR2HSV)
    elif value == 4: return RGB_to_YCrCb(np.array(image)), 'YCbCr', cv2.cvtColor(imageCV, cv2.COLOR_BGR2YCrCb)
    elif value == 5: return HSV_to_RGB(image), 'RGB', cv2.cvtColor(imageCV, cv2.COLOR_HSV2BGR)
    elif value == 6: return YCrCb_to_RGB(image), 'RGB', cv2.cvtColor(imageCV, cv2.COLOR_YCrCb2BGR)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def histTreatments(image, imageCV, Value):
    if Value == 1: return Hist(image), cv2.calcHist([cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    elif Value == 2: return Hist_RGB(image), (cv2.calcHist([imageCV], [0], None, [256], [0, 256]), cv2.calcHist([imageCV], [1], None, [256], [0, 256]), cv2.calcHist([imageCV], [2], None, [256], [0, 256]))
    elif Value == 3: return Cumulative_Hist(Hist(image)), Cumulative_Hist(cv2.calcHist([cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def TransformationsGeo(image, imageCV, x, y, angle, factor, Value):
    if factor < 0: factorCV = 1/abs(factor - 1)
    else: factorCV = factor
    if Value == 1: return Translation(image, int(y), int(x)), cv2.warpAffine(imageCV, np.float32([[1, 0, x],[0, 1, y]]), (imageCV.shape[1], imageCV.shape[0])), 'Translation'
    elif Value == 2: return rotation(image, angle), cv2.warpAffine(imageCV, cv2.getRotationMatrix2D((imageCV.shape[1]/2, imageCV.shape[0]/2), angle, 1), (imageCV.shape[1], imageCV.shape[0])), 'Rotation'
    elif Value == 3: return resize(image, factor), cv2.resize(imageCV, None, fx=factorCV, fy=factorCV, interpolation = cv2.INTER_CUBIC), 'Scale'
    elif Value == 4: return Shear(image, float(y), float(x)), cv2.warpPerspective(imageCV, np.float32([[1, float(x), 0],[float(y), 1, 0],[0, 0, 1]]), (imageCV.shape[1], imageCV.shape[0])), 'shear'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ContourDetection(image, imageCV, value):
    if value == 1: return Naif_Detector(np.array(image)), cv2.add(cv2.filter2D(cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1], -1, filters('naif_x')), cv2.filter2D(cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1], -1, filters('naif_y'))), 'Naif'
    elif value == 2: return Sobel(RGB_to_GRAY(np.array(image))), cv2.add(cv2.Sobel(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)), 'Sobel'
    elif value == 3: return Roberts(np.array(image)), cv2.add(cv2.filter2D(cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1], -1, np.array([[1, 0], [0, -1]])), cv2.filter2D(cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1], -1, np.array([[0, 1], [-1, 0]]))), 'Roberts'
    elif value == 4: return Prewitt(RGB_to_GRAY(np.array(image))), cv2.add(cv2.filter2D(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), -1, np.array([[1, 0, -1],[1, 0, -1], [1, 0, -1] ])) ,cv2.filter2D(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), -1, np.array([[1, 1, 1],[ 0,  0,  0],[ -1,  -1,  -1]]))), 'Prewitt'
    elif value == 5: return convolution(RGB_to_GRAY(np.array(image)), filters('laplacian_4'), 1), cv2.Laplacian(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), cv2.CV_64F), 'laplacian_4'
    elif value == 6: return convolution(RGB_to_GRAY(np.array(image)), filters('laplacian_8'), 1), cv2.filter2D(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), -1, filters('laplacian_8')), 'laplacian_8'
    elif value == 7: return LoG(RGB_to_GRAY(np.array(image))), LoG_CV(imageCV), 'LoG'
    elif value == 8: return DoG(RGB_to_GRAY(np.array(image))), DoG_CV(imageCV), 'DoG'
    elif value == 9: return Canny(np.array(image)), Canny_CV(imageCV), 'Canny'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Smoothing(image, imageCV, value):
    if value == 1: return convolution(np.array(image), filters('moyen'), 1), cv2.filter2D(imageCV,-1, filters('moyen')), 'moyen'
    elif value == 2: return convolution(np.array(image), filters('gaussian_4'), 1), cv2.filter2D(imageCV,-1, filters('gaussian_4')), 'gaussian_4'
    elif value == 3: return convolution(np.array(image), filters('gaussian_8'), 1), cv2.filter2D(imageCV,-1, filters('gaussian_8')), 'gaussian_8'
    elif value == 4: return convolution(np.array(image), filters('gaussian_3x3'), 1), cv2.GaussianBlur(imageCV, (3, 3), 0), 'gaussian_3x3'    
    elif value == 5: return convolution(np.array(image), filters('gaussian_5x5'), 2), cv2.GaussianBlur(imageCV, (5, 5), 0), 'gaussian_5x5' 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Threshold(image, imageCV, value, mn, mx):
    if value == 1: return RGB_to_BIN(np.array(image), mn, mx), cv2.threshold(cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY), mn ,mx, cv2.THRESH_BINARY)[1], 'Threshold'
    elif value == 2: return Hysteresis(RGB_to_GRAY(np.array(image)), mn, mx), imageCV, 'Hysteresis'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Hough(image, imageCV, value, threshold, minRadius, maxRadius):
    if value == 1: return HoughLines(imageCV, threshold), HoughLines_CV(imageCV, threshold), 'HoughLines' #HoughLines(np.array(image))
    elif value == 2: return HoughCircels(imageCV, minRadius, maxRadius, threshold), HoughCircels_CV(imageCV, minRadius, maxRadius), 'HoughCircles'
