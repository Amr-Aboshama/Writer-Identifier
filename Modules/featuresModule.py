import numpy as np
from skimage import feature
from preprocessingModule import *

def LocalBinaryPatternHistogram(numPoints,radius,image):

    lbp = feature.local_binary_pattern(image, numPoints,radius, method="nri_uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0,numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    histSum = hist.sum()
    if histSum != 0:         # Divide if has value
        hist /= histSum
    # return the histogram of Local Binary Patterns
    return hist

#extract Features from each line
def getFeatures(extractedLines, featuresCount):
    interword=[]
    hist=[]
    for img in extractedLines:
        
        hist.append(LocalBinaryPatternHistogram(8,2,img))
    
    return hist
    
# This function is used to return features vectors and their labels
# def getFeaturesAndLabels(image):
#     extractedLines = preprocessImage(image)
#     featuresVectors = np.array(getFeatures(extractedLines))
#     print(featuresVectors.shape)
#     formsFeaturesVectors = np.vstack((formsFeaturesVectors, featuresVectors))
#     for i in range(featuresVectors.shape[0]):
#         labels.append(labelVal)
#     return featuresVectors
