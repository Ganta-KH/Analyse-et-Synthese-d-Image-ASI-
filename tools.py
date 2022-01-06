import numpy as np
from PIL import Image
import cv2

def getImage(image_path):    # obtenir les pixels de la matrice 
    im = Image.open(image_path)   # ouvrir l'image 
    im_rgb = im.convert("RGB")   # le convertir en RGB 
    pixel = np.array(im_rgb)   # RGB matrice
    pixel = pixel.tolist()
    return pixel    # return la matrice

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SaveImage(image, imageCV, name):
    Image.fromarray(np.array(image).astype('uint8')).save('Images/Saved/'+name+'.png', format='png')
    cv2.imwrite('Images/SavedCV/'+name+'.png', imageCV)
    return 'Images/Saved/'+name+'.png', 'Images/SavedCV/'+name+'.png'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def correct_image(image):
    np.putmask(image, image>255, 255)
    np.putmask(image, image<0, 0)
    return image

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def checkNull(var):
        return 0 if len(var) == 0 else var
        """
        if len(var) == 0: return 0
        else: return var
        """





