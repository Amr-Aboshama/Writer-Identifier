import os

featuresCount = 10 
dataPath = 'data'

dirs = os.listdir(dataPath)

timeFile = open(f'{dataPath}/time.txt', 'w')
resultFile = open(f'{dataPath}/result.txt', 'w')

total = len(dirs)
pos = 0

for dir in dirs:
    t, r, sucess = trainAndTestSample(f'{dataPath}/{dir}', featuresCount)
    pos += sucess
    timeFile.write(t + '\n')
    resultFile.write(r + '\n')
    print('Sample ', dir, ': ', sucess)

print('Accuracy = ', pos*100/total, '%')