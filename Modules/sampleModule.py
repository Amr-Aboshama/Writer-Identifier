from timeit import default_timer as timer
from trainingModule import *
import glob
import os

def readData(dataPath):
    trainingImages = []
    trainingLabels = []
    testImages = []
    testLabels = []
    for trainingPath, trainingWriterFolder in enumerate(sorted(glob.glob(dataPath + "/*/"))):
        trainingFiles = sorted(glob.glob(trainingWriterFolder + "/*.PNG"))
        trainingImages += trainingFiles

        trainingLabels += [trainingPath]*len(trainingFiles)

    testImages = sorted(glob.glob(dataPath + "/*.PNG"))


    for i in range(0, len(trainingImages)):
        trainingImages[i] = cv2.imread(trainingImages[i])

    for i in range(0, len(testImages)):
        label = os.path.splitext(os.path.basename(testImages[i]))[0]
        if label.isdigit():
            label = int(label)
        else:
            label = 0
        testLabels.append(label)
        testImages[i] = cv2.imread(testImages[i])

    return trainingImages, trainingLabels, testImages, testLabels


def trainAndTestSample(dataPath, featuresCount):

    # print('Loading Dataset and Labels...')
    trainingImages, trainingLabels, testImages, testLabels = readData(dataPath)
    # print('Finished Loading Dataset and Labels!')

    t0 = timer()
    
    # print('Preprocessing and Feature Extraction from Training Data...')
    xTrain, yTrain = getFeaturesAndLables(trainingImages, trainingLabels, featuresCount)
    # print('Finished Preprocessing and Feature Extraction from Training Data!')
    # print('Preprocessing and Feature Extraction from Test Data...')
    xTest, yTest = getFeaturesAndLables(testImages, testLabels, featuresCount)
    # print('Finished Preprocessing and Feature Extraction from Test Data!')
    
    classification, positive = SVM(xTrain, yTrain, xTest, yTest)
    
    t1 = timer()

    return round(t1-t0, 2), classification, positive
