import numpy as np

def Thresholding(image, mn, mx, mnTre=0.3, mxTre=0.9):
    high = image.max() * mxTre
    low = high * mnTre

    W, H = image.shape

    newImg = np.zeros((W, H))

    mx_i, mx_j = np.where(image >= high)

    mn_i, mn_j =np.where((image <= high) & (image >= low))

    newImg[mx_i, mx_j] = mx
    newImg[mn_i, mn_j] = mn

    return newImg

def Hysteresis(image, mn, mx):
    W, H = image.shape
    for i in range(1, W-1):
        for j in range(1, H-1):
            if image[i, j] == mn:
                k = image[i-1 : i+2, j-1 : j+2]
                if any([[ b == mx for b in a ] for a in k ]):
                    image[i, j] = mx
                else: image[i, j] = 0
    return image
