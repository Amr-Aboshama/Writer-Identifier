from featuresModule import *
from preprocessingModule import *
from sklearn import svm

# This function returns training features and corresponding labels of all test case
def getFeaturesAndLables(images, labels, featuresCount):
    
    xTrain = np.empty((0, featuresCount))
    yTrain = np.empty((0,0))
    for trainingIndex in range(0, len(images)):

        extractedLines,gray = preprocessImage(images[trainingIndex])
        featuresVectors = np.array(getFeatures(extractedLines,gray, featuresCount))
        xTrain = np.vstack((xTrain, featuresVectors))
        yTrain = np.append(yTrain, np.full((featuresVectors.shape[0], 1), labels[trainingIndex]))
    
    return xTrain, yTrain


# This function applies SVM classifier on test features given training set
def SVM(xTrain, yTrain, xTest):
    # print('Training SVM Model...')
    # clf = svm.SVC(gamma='scale')
    clf = svm.SVC(gamma='auto', probability=True, C=5.0)
    clf.fit(xTrain, yTrain)
    # print('Finished Training SVM Model...')
    # print('Predecting Test Results...')
    predictions = clf.predict(xTest)
    # print('Finished Predecting Test Results...')
    uniquePredictions, uniquePredictionsCount = np.unique(predictions, return_counts=True)
    classification = uniquePredictions[np.argmax(uniquePredictionsCount)]
    return int(classification)