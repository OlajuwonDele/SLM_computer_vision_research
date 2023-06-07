import glob
import imutils
import cv2 as cv
import numpy as np
import os, os.path
from scipy import ndimage
from random import randint
import matplotlib.cm as cm 
from functools import reduce
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


#from skimage import io

x_list = []
y_list = []
shape_array = []
image_array = [] 
velocity_list = []
u_vector_list = []
v_vector_list = []

class Shape:
    position_array = []
    u_vector_array = []
    v_vector_array = []

    velocity = u_vector = v_vector = 0
    is_black = False #if true, then it's a grey circle
    
    def __init__(self, x, y, radius, number, i, b, g ,r):
        self.x = x
        self.y = y
        self.radius = radius
        self.position_array.append((x,y))
        self.number = number
        self.i = i
        self.b = b
        self.g = g
        self.r = r

    def positionAppend(self, x, y, i):
        self.u_vector = (x - self.x) / 10
        self.u_vector_array.append(self.u_vector)
        self.v_vector = (y - self.y) / 10
        self.v_vector_array.append(self.v_vector)
        self.velocity = round(np.sqrt(self.u_vector**2 + self.v_vector**2), 3) #would divide by the time between images rather than two, maths -- getting magnitude of velocity in x and y
        self.position_array.append((x,y))
        self.x = x
        self.y = y
        self.i = i
      
def velocityProfile(x,y,velocity,u, v):
    x_list.append(x)
    y_list.append(y)
    velocity_list.append(velocity)
    u_vector_list.append(u)
    v_vector_list.append(v)

def get_intersections(x0, y0, r0, x1, y1, r1, d):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
        a=(r0**2-r1**2+d**2)/(2*d)
        h=np.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (x3, y3, x4, y4)

def ternization(img, image):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #need to intergrate ternization method later from other file into this main project
    #this method will update status of shapes to either being classified as a black shape or grey shape
    pass

def watershedImageFilter(img):
    blur = cv.GaussianBlur(img,(7,7),0) #applies a Guassian filter to the image
    #converts BGR color space change to HSV
    mSource_Hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV) 
    mMask = cv.inRange(mSource_Hsv,np.array([0,0,40]),np.array([100,130,255]));
    output = cv.bitwise_and(img, img, mask=mMask)

    #grayscale
    img_grey = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

    #thresholding
    ret,th1 = cv.threshold(img_grey,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

    #dist transform
    dist = ndimage.distance_transform_edt(th1)

    #markers
    localMax = peak_local_max(dist, indices=False, min_distance=20, labels=th1)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    #apply watershed
    labels = watershed(-dist, markers, mask=th1)
    return th1, labels, img_grey

def watershedTracker(i, contourNum, last_contour, img, img_grey, labels, contours):
    # loop over the unique labels
    same_shape = False
    for label in np.unique(labels):
        if label == 0:
            continue

        # draw label on the mask
        mask = np.zeros(img_grey.shape, dtype="uint8")
        mask[labels == label] = 255
   
        # detect contours in the mask and grab the largest one
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        ((x, y), radius) = cv.minEnclosingCircle(c)
        x = int(x)
        y = int(y)
        radius = int(radius)
        
        #adds new detected shape to our database
        if i == 0: 
            shape_array.append(Shape(x, y, radius, contourNum, i, randint(100,255),randint(100,255),randint(100,255))) 
            contourNum += 1 
            # makes new unique shape identifier number

        # this is after the first image
        else: 
            for obj in shape_array:
                if (obj.radius - 1 <= radius <= obj.radius + 1):
                    print(np.sqrt((shape_array[obj.number].x - x)**2 + (shape_array[obj.number].y - y)**2))
                    same_shape = True
                    break
            #if area is the same but the position has changed
            if same_shape == True and (x != obj.x or y != obj.y): 
                #print('Shape Number: ' + str(obj.number) + ' has moved from (' + str(obj.x) + ', ' + str(obj.y) + ') to (' + str(x) + ', ' + str(y) + ')')
                #cv.drawContours(img, last_contour, -1, (obj.b-50,obj.g-50,obj.r-50), thickness = 2) 
                #cv.arrowedLine(img, (obj.x,obj.y), (x, y),
                #                     (0,0,0), 1)
                shape_array[obj.number].positionAppend(x, y, i)
                #adds new changes of shape xx to database and stores all previous information too
            if same_shape == True and (int(x) == obj.x and int(y) == obj.y):
                #print('same shape but same position, need to do something for this')
                pass
            #new shape not previously detected
            else:
                shape_array.append(Shape(x, y, radius, contourNum, i, randint(50,255),randint(100,255),randint(100,255))) #adds new detected shape to our database
                contourNum += 1 # makes new unique shape identifier number
                #print('not the same')
     
    
        cv.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        # cv.drawContours(img, cnts, -1, (0, 0, 0), 2)
        cv.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 1)
        # cv.circle(img, (int(x), int(y)), 5, (255, 0, 0), thickness=cv.FILLED)
        #cv.drawContours(img, last_contour, -1, (obj.b-50,obj.g-50,obj.r-50), thickness = 2)

    for index, main_shape, in enumerate(shape_array):
        cv.circle(img, (main_shape.x, main_shape.y), 5, (main_shape.b-50, main_shape.g-50 , main_shape.r-50), -1) #draws the centres of each object
        #cv.drawContours(img, contours, -1, (0,0,0), thickness=2) #draws the outlines of each object
        #text = 'Position: ' + str(main_shape.i) + ', Shape ' + str(main_shape.number) + ', Velocity: ' + str(main_shape.velocity) + ' m/s'
        #cv.putText(img,text, (main_shape.x - 20, main_shape.y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (main_shape.b-50, main_shape.g-50 ,main_shape.r-50), 2) #puts text onto image
        print('Shape Number: ' + str(main_shape.number) + ' Radius: ' + str(main_shape.radius) + ' X: '+ str(main_shape.x) + ' Y: ' + str(main_shape.y))

        for iter in range(index + 1, len(shape_array)):
            d = np.sqrt((shape_array[iter].x - main_shape.x)**2 + (shape_array[iter].y - main_shape.y)**2)
            # Check if circles are non intersecting or if one circle within other or if coincident circles
            if (d > main_shape.radius + shape_array[iter].radius) or (d < abs(main_shape.radius - shape_array[iter].radius)) or (d == 0 and main_shape.radius == shape_array[iter].radius):
                pass
            else:
                x1,y1,x2,y2 = get_intersections(main_shape.x, main_shape.y, main_shape.radius, shape_array[iter].x, shape_array[iter].y, shape_array[iter].radius, d)
                cv.circle(img, (int(x1), int(y1)), 5, (255, 255, 255), thickness=cv.FILLED)
                cv.circle(img, (int(x2), int(y2)), 5, (255, 255, 255), thickness=cv.FILLED)
        
    return contourNum, img
    
def contourTracker(contours, i, contourNum, last_contour, img):
    same_shape = False
    for c in contours:
        M = cv.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) # m10 = sum of x, m00 = area of moment # m01 = sum of y
            cY = int(M["m01"] / M["m00"]) 
        else:
            cX, cY = 0, 0
        
        if i == 0: #this is when we are on the very first image
            shape_array.append(Shape(cX, cY, M["m00"], M["m10"], contourNum, i, randint(100,255),randint(100,255),randint(100,255))) #adds new detected shape to our database
            contourNum += 1 # makes new unique shape identifier number

        else: # this is after the first image
            for obj in shape_array: #cycles through the objects already stored in our shape list
                if obj.area - 30 < M["m00"] < obj.area + 30: #need to get std deviation for area and position change for this bit would prefer to use machine learning in the future,
                    #could also useful to calc shortest distance from centroid to edge for more verification as this may lead to future problems
                    same_shape = True
                    break
            if same_shape == True and (cX != obj.x or cY != obj.y): #if area is the same but the position has changed
                print('Shape Number: ' + str(obj.number) + ' has moved from (' + str(obj.x) + ', ' + str(obj.y) + ') to (' + str(cX) + ', ' + str(cY) + ')')
                cv.drawContours(img, last_contour, -1, (obj.b-50,obj.g-50,obj.r-50), thickness = 2) #-1 draws all contours of the previous image to show movement,
                cv.arrowedLine(img, (obj.x,obj.y), (cX, cY),
                                     (0,0,0), 1)
                shape_array[obj.number].positionAppend(cX, cY, i)
                #adds new changes of shape xx to database and stores all previous information too
            if same_shape == True and (cX == obj.x and cY == obj.y):
                #print('same shape but same position, need to do something for this')
                pass
            else:
                shape_array.append(Shape(cX, cY, M["m00"], M["m10"], contourNum, i, randint(50,255),randint(100,255),randint(100,255))) #adds new detected shape to our database
                contourNum += 1 # makes new unique shape identifier number
                #print('not the same')
          

    for obj in shape_array: #loop to draw each shape in the current image 
        if obj.i == i:
            cv.circle(img, (obj.x, obj.y), 5, (obj.b-50,obj.g-50 ,obj.r-50), -1) #draws the centres of each object
            cv.drawContours(img, contours, -1, (0,0,0), thickness=2) #draws the outlines of each object
            text = 'Position: ' + str(obj.i) + ', Shape ' + str(obj.number) + ', Velocity: ' + str(obj.velocity) + ' m/s'
            cv.putText(img,text, (obj.x - 20, obj.y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (obj.b-50,obj.g-50 ,obj.r-50), 2) #puts text onto image
            print('Shape Number: ' + str(obj.number) + ' Area: ' + str(obj.area) + ' X: '+ str(obj.x) + ' Y: ' + str(obj.y)) #prints results

        velocityProfile(obj.x, obj.y, obj.velocity, obj.u_vector, obj.v_vector)
    
    
    #plt.axis('equal')
    plt.quiver(x_list, y_list, u_vector_list, v_vector_list, velocity_list, scale= 1, units='xy', cmap = 'jet')
    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.xlabel("x/cm")
    plt.ylabel("y/cm")
    plt.title("Vector Field Plot")
    cbar = plt.colorbar()
    cbar.set_label('Velocity (m/s)')
    plt.show()

    return contourNum #returns the current number of shapes saved 

def imageStrengthen(img):
    blur = cv.GaussianBlur(img, (7, 7), 2)
    h, w = blur.shape[:2]
    
    # Morphological gradient
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    gradient = cv.morphologyEx(blur, cv.MORPH_GRADIENT, kernel)
  

    # Binarize gradient
    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv.inRange(gradient, lowerb, upperb)


    # Flood fill from the edges to remove edge crystals
    for row in range(h):
        if binary[row, 0] == 255:
            cv.floodFill(binary, None, (0, row), 0)
        if binary[row, w-1] == 255:
            cv.floodFill(binary, None, (w-1, row), 0)

    for col in range(w):
        if binary[0, col] == 255:
            cv.floodFill(binary, None, (col, 0), 0)
        if binary[h-1, col] == 255:
            cv.floodFill(binary, None, (col, h-1), 0)

    # Cleaning up mask
    foreground = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(foreground, cv.MORPH_CLOSE, kernel)



#filenames = glob.glob("C:/Users/juwon/Downloads/velocity/*.png")
#filenames = glob.glob("C:/Users/juwon/Downloads/imaged/*.png")
filenames = glob.glob("C:/Users/juwon/Downloads/nnn/*.png")
#filenames = glob.glob("C:/Users/juwon/Downloads/wazoo/*jpg")

#filenames = glob.glob("C:/Users/juwon/Downloads/test/*.png")
filenames = glob.glob("C:/Users/juwon/Downloads/test/*.jpg")
filenames.sort() #sorts images in alphabetical order


for file in filenames:
    img = cv.imread(file) #reads current image in filenames loop
    image_array.append(img)#adds image to image array and cycles through the folder of filenames

print('data shape:', np.array(image_array).shape) #prints the shape of data, eg (12, 800, 800, 3), 12 = number of images and width and height of image


contourNum = 0
last_contour = []
for i in range(len(image_array)): #cycles through the size of image_array
    th1, labels, img_grey = watershedImageFilter(image_array[i])
    width, height, _ = img.shape
    img = 255 * np.ones((width, height,3), np.uint8) #creates blank image 
    gray = cv.cvtColor(image_array[i], cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV) #binarization at a specific range of values (close to trinarization method)
    edged = cv.Canny(gray, 30, 200)

    #this is when we are on the first image
    if i == 0:
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #vary edged or thresh
        last_contour = contours
    else:
        last_contour = contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contourNum, image = watershedTracker(i, contourNum, last_contour, image_array[i].copy(), img_grey.copy(), labels, contours)
    x_list.clear()
    y_list.clear()
    velocity_list.clear()
    u_vector_list.clear()
    v_vector_list.clear()
    
    #cv.imshow('Original', image_array[i])
   # cv.imshow('Edit', image)
    #cv.waitKey(0)

    #cv.drawContours(img, contours, -1, (0,0,0), thickness=2)
    plt.subplot(121),plt.imshow(image_array[i])
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(image)
    plt.title('Points of Intersection'), plt.xticks([]), plt.yticks([]) #creates plot of images, colour is weird for some reason so I'll fix later.
    plt.show()