import cv2
import numpy as np
from copy import deepcopy

################----------------Preprocessing MODULE--------------------####################
#This function Takes the image and returns its extracted lines
def preprocessModule(filename):
    # Read image and convert it to binary
    img = cv2.imread(filename)

    image,binarizedImage=crop_text(img) 
    gray_img=deepcopy(image)
    image[binarizedImage == 0] = 0
    image[binarizedImage == 255] = 1
  
    verticalHistogram = image.sum(axis=1)
    smallest = int(np.average(verticalHistogram) - np.min(verticalHistogram)) / 4
    
    linesArrays = splitz(verticalHistogram, int(smallest))
   
    horizontalHistogram = image.sum(axis=0)
   
    smallest = int(np.average(horizontalHistogram) - np.min(horizontalHistogram)) / 4
    marginsArrays = (list(splitz(horizontalHistogram[30:], smallest)))

    counter = 0
    extractedLines = []
    extractedLines_gray=[]

    # For each array (representing a line) extracted, perform some preprocessing operations
    for arr in (linesArrays):
        if (arr[-1] - arr[0] > 30):
            line = image[arr[0]:arr[-1], marginsArrays[0][0]:marginsArrays[-1][-1]]
            line[line != 0] = 255
            extractedLines.append(line)
            
            line_gray = gray_img[arr[0]:arr[-1], marginsArrays[0][0]:marginsArrays[-1][-1]]
            extractedLines_gray.append(line_gray)
            
            counter += 1

    return extractedLines,extractedLines_gray

#This function extracts the text part only from IAM image 
def crop_text(image):
   # Load image, convert to grayscale, Otsu's threshold
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
 

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #upper horizontal
    x,y,w,h = cv2.boundingRect(cnts[-2])

    #lower horizontal
    x1,y1,w1,h1 = cv2.boundingRect(cnts[0])
  
    #final text only extract
    gray=gray[y+h+5:y1,:]

    #done cropping from horizontal lines
    #continue to remove extra white space

    # Create rectangular structuring element and dilate
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    dilate = cv2.dilate(thresh[y+h+5:y1,:], kernel, iterations=4)
 
    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        largest = max(cnts, key = cv2.contourArea)

    xf,yf,wf,hf = cv2.boundingRect(largest)
    cv2.rectangle(blur,(xf,yf),(xf+wf,yf+hf),(0,255,0),2) 
    prepro_img=blur[yf:yf+hf, xf:xf+wf]
    thresh = cv2.threshold(prepro_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    return prepro_img,thresh

# This function splits a sequence of numbers on a given value (smallest)
def splitz(seq, smallest):
    group = []
    for i in range(len(seq)):
        if (seq[i] >= (smallest)):
            group.append(i)
        elif group:
            yield group
            group = []

#test for preprocessing Module
# extractedLines,extractedLines_gray=preprocessModule("Test Samples/data/01/1.png")
# print(extractedLines[0].shape)

# for line in extractedLines:
#     plt.imshow(line,cmap='gray')
#     plt.show()