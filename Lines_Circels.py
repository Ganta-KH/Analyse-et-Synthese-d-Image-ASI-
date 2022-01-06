import numpy as np
import cv2

def HoughLines(image, threshold, mn=100, mx=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, mn, mx)

    W, H = canny.shape
    Dmax = int(np.ceil(np.sqrt(W ** 2 + H ** 2)))

    houghSpace = np.zeros((Dmax * 2, 180), dtype=np.uint64)
    Ys, Xs = np.nonzero(canny)

    thetas = np.arange(180)
    thetas_x = np.cos(thetas)
    thetas_y = np.sin(thetas)

    for x, y in zip(Xs, Ys):
        rho = np.ceil(x * thetas_x + y * thetas_y).astype(int) + Dmax
        houghSpace[rho, thetas] += 1

    # draw lines
    rhos, thetas = np.where(houghSpace >= threshold)
    if rhos is not None and thetas is not None:
        rhos -= Dmax
        for rho, theta in zip(rhos, thetas):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            point1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            point2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, point1, point2, (0, 255, 0), 2)

    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def HoughLines_CV(image, threshold, mn=100, mx=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, mn, mx)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, threshold)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            point1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            point2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, point1, point2, (0, 255, 0), 2)
    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def HoughCircels(image, minRadius, maxRadius, threshold, mn=100, mx=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, mn, mx)

    W, H = canny.shape
    Ys, Xs = np.nonzero(canny)

    thetas= np.linspace(0,2*np.pi, 250)
    thetas_x = np.cos(thetas)
    thetas_y = np.sin(thetas)
    Dmax = int(np.ceil(np.sqrt(W ** 2 + H ** 2)))
    houghSpace = np.zeros((W+Dmax, H+Dmax), dtype=np.uint64)

    radius = np.arange(minRadius, maxRadius)
    #Rx = np.array(list(map(lambda t: t * radius, thetas_x)))
    #Ry = np.array(list(map(lambda t: t * radius, thetas_y)))

    for radius in range(minRadius, maxRadius):
        Rx = radius * thetas_x
        Ry = radius * thetas_y
        for centerX, centerY in zip(Xs, Ys):
            x = (Rx + centerX).astype(int)
            y = (Ry + centerY).astype(int)
            houghSpace[x + (Dmax//2), y + (Dmax//2)] += 1

        xs, ys = np.where(houghSpace >= threshold) # get the circels points
        xs -= Dmax//2
        ys -= Dmax//2

        # draw circels
        for x, y in zip(xs, ys):
            cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

        houghSpace[...] = 0

    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def HoughCircels_CV(image, minRadius, maxRadius,mn=100, mx=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    circels = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=mx, param2=mn, minRadius=minRadius, maxRadius=maxRadius)

    if circels is not None:
        circels = np.uint16(np.around(circels))
        for x, y, radius in circels[0, :]:
            cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
        
    return image