import cv2
import numpy as np
from matplotlib import pyplot as plt

#This function Takes the image and returns its extracted lines
def preprocessImage(img, scale_perc= 0.45):
    # Convert image to binary
    # plt.imshow(img,cmap='gray')
    # plt.show()
    
    h = int(img.shape[0] * scale_perc)
    w = int(img.shape[1] * scale_perc)
    img = cv2.resize(img, (w, h))

    gray_img,binarizedImage=crop_text(img, scale_perc) 
    
    
    # this array contains summation of all *black* pixels on each row of the image
    row_hist = binarizedImage.sum(axis=1)
   
   
    is_lines = row_hist > 5
    
    lines = []
    gray_lines=[]
    i = 0
    while i < len(is_lines):
        if is_lines[i]:
            begin_row = i
            lower_bound = max(begin_row - 5, 0)
            while i < len(is_lines) and is_lines[i]:
                i += 1
            upper_bound = min(i + 5, len(is_lines) - 1)
            if i - begin_row > 50 *scale_perc:  # threshold for # of rows to be higher than 20 row
                lines.append(binarizedImage[lower_bound:upper_bound, :])
                gray_lines.append(gray_img[lower_bound:upper_bound, :])
        i += 1
    return lines,gray_lines

#This function extracts the text part only from IAM image 
def crop_text(image, scale_perc):
   # Load image, convert to grayscale, Otsu's threshold
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
 

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(scale_perc*80),1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #upper horizontal
    x,y,w,h = cv2.boundingRect(cnts[-2])

    #lower horizontal
    x1,y1,w1,h1 = cv2.boundingRect(cnts[0])
  
    #final text only extract
    gray=gray[y+h+int(scale_perc*5):y1,:]

    #done cropping from horizontal lines
    #continue to remove extra white space

    # Create rectangular structuring element and dilate
    blur = cv2.GaussianBlur(gray, (int(scale_perc*7),int(scale_perc*7)), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(scale_perc*50),int(scale_perc*50)))
    dilate = cv2.dilate(thresh[y+h+int(scale_perc*5):y1,:], kernel, iterations=4)
    largest=0

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

    thresh[thresh == 255] = 1
    return prepro_img,thresh



#test for preprocessing Module
# extractedLines,extractedLines_gray=preprocessModule("Test Samples/data/01/1.png")
# print(extractedLines[0].shape)

# for line in extractedLines:
#     plt.imshow(line,cmap='gray')
#     plt.show()