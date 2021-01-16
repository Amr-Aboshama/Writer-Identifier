import numpy as np
import glob
import time

#######-------------------------MAIN---------------------------######
##Global variables

featuresCount = 10 

# create output files
f = open("results.txt", "w+")
f.close()
f = open("time.txt", "w+")
f.close()

for TestFolder in sorted(glob.glob("data/*")):
    
    trainingImagePaths, testingImagePath, trainingLabels = readData(TestFolder)
   
    t0 = time.time()
    
    xTrain = np.empty([0, featuresCount])
    yTrain = []
    xTest = np.empty([0, featuresCount])
    featuresVectors = []
    ImageFeaturesVectors = np.empty([0, featuresCount])
    testFeaturesVectors = np.empty([0, featuresCount])
    
    
    xTrain, yTrain = getTrainingData(trainingImagePaths, trainingLabels, yTrain, ImageFeaturesVectors)
    
    xTest= getTestFeatures(testingImagePath, testFeaturesVectors)
    
    classification, predictions=SVM(xTrain, yTrain, xTest)
    
    t1 = time.time()
    writeOutput(classification, t1, t0)
