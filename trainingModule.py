import numpy as np

###################--------------GET TRAINING DATA------------------#####################
# This function returns training features and corresponding labels of all test case
def getTrainingData(trainingFormIDs, trainingLabels, yTrain, formsFeaturesVectors):
    
    for trainingIndex, formID in enumerate(trainingFormIDs):
        formsFeaturesVectors, labels= getFeaturesAndLabels(formID, trainingLabels[trainingIndex], formsFeaturesVectors, yTrain)

    xTrain = np.array(formsFeaturesVectors)
    yTrain = np.array(yTrain)
    
    return xTrain, yTrain

# This function is used to return test features vectors
def getTestFeatures(filename, formsFeaturesVectors):
    print("current image is ", filename)
    extractedLines,gray = preprocessModule(filename)

    featuresVectors = np.array(getFeatures(extractedLines,gray))
        
    formsFeaturesVectors = np.vstack((formsFeaturesVectors, featuresVectors))
            
    return np.array(formsFeaturesVectors)

##############----------------------MODEL-------------------#############
# This function applies SVM classifier on test features given training set
def SVM(xTrain, yTrain, xTest):
    
    clf = svm.SVC(gamma='scale')
    clf.fit(xTrain, yTrain)
    predictions = clf.predict(xTest)

    uniquePredictions, uniquePredictionsCount = np.unique(predictions, return_counts=True)
    classification = uniquePredictions[np.argmax(uniquePredictionsCount)]
    return classification, predictions