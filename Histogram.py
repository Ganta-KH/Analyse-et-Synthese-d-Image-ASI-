import numpy as np
from ConversionImage import RGB_to_GRAY
from itertools import accumulate

def Hist(image):
    image = RGB_to_GRAY(np.array(image))
    hist = np.zeros(256)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    return hist

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Hist_RGB(image):
    histR = np.zeros(256)
    histG = np.zeros(256)
    histB = np.zeros(256)

    for i in range(len(image)):
        for j in range(len(image[0])):
            histR[image[i][j][0]] += 1
            histG[image[i][j][1]] += 1
            histB[image[i][j][2]] += 1
    return (histR, histG, histB)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Cumulative_Hist(hist):
    return np.array( list( accumulate( hist ) ) )
    """
    cumuHist = np.zeros(256)
    cumuHist.itemset(0, hist.item(0))
    for i in range(1, 256):
        cumuHist.itemset(i, hist.item(i) + cumuHist.item(i-1))
    return cumuHist
    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def equalization_Hist(image):
    hist = Hist(image) # calculate histogram
    image = RGB_to_GRAY(np.array(image))
    flat = image.flatten()

    cumuHist = Cumulative_Hist(hist) # calculate Cumu

    nj = (cumuHist - cumuHist.min()) * 255 # normaliser enter 0 - 255
    N = cumuHist.max() - cumuHist.min()
    cumuHist = nj / N
    cumuHist = cumuHist.astype('uint8')
    newImg = cumuHist[flat]
    newImg = np.reshape(newImg, image.shape)
    return newImg
