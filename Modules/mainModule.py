import os
from sampleModule import *

featuresCount = 10 
dataPath = 'data2'

dirs = os.listdir(dataPath)

timeFile = open(f'{dataPath}/time.txt', 'w')
resultFile = open(f'{dataPath}/result.txt', 'w')

total = 0
pos = 0
time = 0

for dir in dirs:
    if not os.path.isdir(os.path.join(dataPath, dir)):
        continue

    total += 1
    t, r, sucess = trainAndTestSample(f'{dataPath}\\{dir}', featuresCount)
    pos += sucess
    time += t
    timeFile.write(f'{t}\n')
    resultFile.write(f'{r}\n')
    print('Sample ', dir, ': ', sucess)

print('Accuracy = ', pos*100/total, '%')
print('Total Time = ', time, ' seconds')