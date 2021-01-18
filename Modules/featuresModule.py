import numpy as np
from skimage import feature
from preprocessingModule import *
import cv2

def LocalBinaryPatternHistogram(image,mask):
    hist = np.zeros(256)
    #mask=255-mask
    lbp = np.uint8(feature.local_binary_pattern(image, 8,1))
    #print(lbp)
    # Calculate LBP histogram of only black pixels
    hist = cv2.calcHist([lbp], [0], mask, [256], [0, 256], hist, True)
    hist = hist.ravel()
    #ist /= np.sum(hist)
  
    return hist

#extract Features from each line
# def getFeatures(extractedLines, featuresCount):
#     interword=[]
#     hist=[]
#     for img in extractedLines:
        
#         hist.append(LocalBinaryPatternHistogram(8,2,img))
    
#     return hist

def getFeatures(extractedLines,extractedLines_gray, featuresCount):
    hist=[]
    for i,j in zip(extractedLines,extractedLines_gray):
        hist.append(LocalBinaryPatternHistogram(j,i))

    #print(hist)
    hist /=np.mean(hist)
    return hist
    