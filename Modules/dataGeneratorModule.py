import os
import sys
import random
from shutil import copyfile, rmtree

if len(sys.argv) != 4:
    print('Error in arguments. It expects 3 arguments: <Input-Directory-Path> <Out-Directory-Path> <Documents-Count>')
    exit()

inputDirectoryPath = sys.argv[1]
outDirectoryPath = sys.argv[2]
docs = int(sys.argv[3])

formsPath = f'{inputDirectoryPath}/forms.txt'
imagesPath = f'{inputDirectoryPath}/images'

formsFile = open(formsPath, 'r')

formsLines = formsFile.readlines()

writers = dict()
wList = list()

for line in formsLines:
    if line.startswith('#'):
        continue

    lSplit = line.split(' ')
    writers.setdefault(lSplit[1], list())
    writers[lSplit[1]].append(lSplit[0])

for k, v in writers.items():
    if len(v) <= 2:
        continue
    wList.append((k, v))    


if os.path.exists(outDirectoryPath):
    rmtree(outDirectoryPath)

os.mkdir(outDirectoryPath)

while docs > 0:
    path = f'{outDirectoryPath}/{docs:04d}' 
    os.mkdir(path)
    w = random.sample(wList, 3)
    k = random.randint(0, 2)

    for i in range(0, len(w)):
        # print(w[i][0])
        os.mkdir(f'{path}/{i+1}')
        d = None

        if k==i:
            d = random.sample(w[i][1], 3)
            # print(f'test: {d[-1]}')
            copyfile(f'{imagesPath}/{d.pop()}.png', f'{path}/{i+1}.png')
        else:
            d = random.sample(w[i][1], 2)
        
        while len(d):
            # print(f'{outDirectoryPath}[{len(d)}]: {d[-1]}')
            copyfile(f'{imagesPath}/{d.pop()}.png', f'{path}/{i+1}/{len(d)+1}.png')

    docs -= 1