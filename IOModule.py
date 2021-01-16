import glob

#############----------------------IO function-----------------#############
# This function reads forms from folders representing each iteration inside folder data
def readData(folderName):

    trainingImagePaths = []
    trainingLabels = []
    for trainingPath, trainingWriterFolder in enumerate(sorted(glob.glob(folderName + "/*/"))):
      
        trainingFiles = sorted(glob.glob(trainingWriterFolder + "/*.PNG"))
        trainingImagePaths += trainingFiles

        trainingLabels += [trainingPath]*len(trainingFiles)

    testingFormPath = glob.glob(folderName + "/*.PNG")[0]

    return trainingImagePaths, testingFormPath,trainingLabels

# This function writes the classified class to a results.txt file and the time of training and testing in time.txt file
def writeOutput(classification, t1, t0):
    f = open("results.txt", "a")
    f.write(str(classification + 1) + "\n")
    f.close()

    f = open("time.txt", "a")
    f.write(str(round(t1 - t0, 2)) + "\n")
    f.close()