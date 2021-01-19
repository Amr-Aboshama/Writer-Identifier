import os
import sys
from sampleModule import *

featuresCount = 256


if len(sys.argv) != 3:
    print('Error in arguments. It expects 2 arguments: <Input-Data-Path> <Self-Analysis-Mode(0 inactive, otherwise active)>')
    exit()

dataPath = sys.argv[1]
selfAnalysisMode = int(sys.argv[2])

dirs = os.listdir(dataPath)

timeFile = open(f'{dataPath}/time.txt', 'w')
resultFile = open(f'{dataPath}/results.txt', 'w')

total = 0
pos = 0
time = 0

for dir in dirs:
    if not os.path.isdir(os.path.join(dataPath, dir)):
        continue
    
    total += 1
    t, r, positive = trainAndTestSample(f'{dataPath}\\{dir}', featuresCount)
    pos += positive
    time += t
    timeFile.write(f'{t}\n')
    resultFile.write(f'{r}\n')
    if selfAnalysisMode:
        print('Sample ', dir, ': ', positive)
    else:
        print('Sample ', dir, ': Done')

if selfAnalysisMode:
    print('Accuracy = ', pos*100/total, '%')
print('Total Time = ', time, ' seconds')