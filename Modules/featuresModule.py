import numpy as np
from skimage import feature
from preprocessingModule import *
import cv2

def LocalBinaryPatternHistogram(image,mask):
    hist = np.zeros(256)
    
    lbp = np.uint8(feature.local_binary_pattern(image, 8,1))
    
    # Calculate LBP histogram 
    #(images: lbp,channel: grayscale,mask: calculate only text part , histsize ,histrange,hist)
    hist = cv2.calcHist([lbp], [0], mask, [256], [0, 256], hist)
    
    #return 1D list
    hist = hist.ravel()
    
    return hist


def getFeatures(extractedLines,extractedLines_gray, featuresCount):
    hist=[]
    for i,j in zip(extractedLines,extractedLines_gray):
        hist.append(LocalBinaryPatternHistogram(j,i))

    
    hist /=np.mean(hist)
    return hist
    