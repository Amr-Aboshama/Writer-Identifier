import numpy as np
from skimage import feature

def LocalBinaryPatternHistogram(numPoints,radius,image):

    lbp = feature.local_binary_pattern(image, numPoints,radius, method="nri_uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0,numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum())
    # return the histogram of Local Binary Patterns
    return hist

#extract Features from each line
def getFeatures(extractedLines,extractedLines_gray):
    interword=[]
    hist=[]
    for i,j in zip(extractedLines,extractedLines_gray):
        
        hist.append(LocalBinaryPatternHistogram(8,2,i))
    
    return hist
    
# This function is used to return features vectors and their labels
def getFeaturesAndLabels(filename, labelVal, formsFeaturesVectors, labels):
    
    print("current filename is ", filename)
   
    extractedLines,gray = preprocessModule(filename)
    
    featuresVectors = np.array(getFeatures(extractedLines,gray))  
    #featuresVectors=featuresVectors.reshape(1,-1)
    
    print(featuresVectors.shape)
    #print(featuresVectors)
    
    formsFeaturesVectors = np.vstack((formsFeaturesVectors, featuresVectors)) 
    
    for i in range(featuresVectors.shape[0]):
        labels.append(labelVal)

    return np.array(formsFeaturesVectors), labels

    