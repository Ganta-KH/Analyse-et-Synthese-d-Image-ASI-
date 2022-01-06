import numpy as np
import math

def Translation(image, x, y):
    if x == y == 0: return image
    h, w = len(image), len(image[0])
    newImg = []
    for i in range(h):
        newImg.append([])
        for j in range(w):
            if (0 <= (i - x) < h) and (0 <= (j - y) < w): newImg[i].append(image[i-x][j-y])
            else: newImg[i].append([0, 0, 0])
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def rotation(image, angle):
    image = np.array(image)

    angle=math.radians(angle) #nous faisons l'entrée de l'utilisateur comme un angle pour que python le comprenne comme ça                             
    cosine=math.cos(angle)# on prend le cos de l'angle 
    sine=math.sin(angle)# on prend le sin de l'angle 
    height=image.shape[0]                                 
    width=image.shape[1]      

    new_height  = round(abs(height*cosine)+abs(width*sine))+1 # nous faisons des calculs ici pour obtenir la nouvelle hauteur d'image 
    new_width  = round(abs(width*cosine)+abs(height*sine))+1 # nous faisons quelques calculs ici pour obtenir la nouvelle largeur de l'image 

    newimage=np.zeros((new_height,new_width,image.shape[2])) #nous remplissons la nouvelle image avec des zéros ... image.shape[2] = 3 cela signifie tuple RVB de l'image et nous les remplissons de zéros 
    
    # Trouver le centre de l'image autour duquel nous devons faire pivoter l'image 
    original_centre_height   = round(((height+1)/2)-1)  
    original_centre_width    = round(((width+1)/2)-1)  
    
    # Trouver le centre de la nouvelle image qui sera obtenue 
    new_centre_height = round(((new_height+1)/2)-1)        
    new_centre_width = round(((new_width+1)/2)-1)       

    for i in range(height):
        for j in range(width):
            #coordonnées du pixel par rapport au centre de l'image originale 
            y=height-1-i-original_centre_height                   
            x=width-1-j-original_centre_width                
            #coordonnée du pixel par rapport à l'image pivotée 
            new_y=round(-x*sine+y*cosine)
            new_x=round(x*cosine+y*sine)
            #puisque l'image sera tournée, le centre changera aussi, donc pour s'adapter à cela, nous devrons changer new_x et new_y par rapport au nouveau centre 
            new_y=new_centre_height-new_y
            new_x=new_centre_width-new_x
            # ajouter if check pour éviter toute erreur dans le traitement 
            if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                newimage[new_y,new_x,:]=image[i,j,:]   #écriture des pixels vers la nouvelle destination dans l'image de sortie                        

    return newimage

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def resize(image_array, factor):
    W, H = len(image_array[0]), len(image_array)
    fac = 0
    if factor == 0 or factor == 1: return image_array
    elif factor > 1: newW, newH = W * factor, H * factor
    else:
        factor = (factor - 1) * -1
        fac = factor
        newW, newH = W//factor, H//factor
        factor = 1/factor

    newImage = np.zeros((newH,newW,3)).tolist()

    for col in range(newW):
        for row in range(newH):
            p = image_array[int(row//factor-fac)][int(col//factor-fac)]
            newImage[row][col] = p

    return newImage
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Shear(image, shx, shy):
    h, w = len(image), len(image[0])
    h2, w2 = h, w
    start_h, start_w = 0, 0
    new_x, new_y = int(h + shx * w), int(shy * h + w)
    if new_x > h-1: h2 += new_x - h
    if new_y > w-1: w2 += new_y - w
    new_x, new_y = int(shx * w), int(shy * h)
    if new_x < 0: h2 += abs(new_x); start_h = abs(new_x)
    if new_y < 0: w2 += abs(new_y); start_w = abs(new_y)

    newImg = np.zeros((h2, w2, 3))
    for x in range(h):
        for y in range(w):
            new_x, new_y = int(x + shx * y), int(shy * x + y)
            newImg[new_x + start_h, new_y + start_w] = image[x][y]

    return newImg
  