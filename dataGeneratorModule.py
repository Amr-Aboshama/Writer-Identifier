import os
import random
from shutil import copyfile, rmtree




docs = 200
outDirectory = 'data2'

formsFile = open('forms.txt', 'r')

formsLines = formsFile.readlines()

writers = dict()
wList = list()

for line in formsLines:
    if line.startswith('#') or line[0]>'d':
        continue

    lSplit = line.split(' ')
    writers.setdefault(lSplit[1], list())
    writers[lSplit[1]].append(lSplit[0])

for k, v in writers.items():
    if len(v) <= 2:
        continue
    wList.append((k, v))    


if os.path.exists(outDirectory):
    rmtree(outDirectory)

os.mkdir(outDirectory)

while docs > 0:
    path = f'{outDirectory}/{docs:04d}' 
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
            copyfile(f'images/{d.pop()}.png', f'{path}/{i+1}.png')
        else:
            d = random.sample(w[i][1], 2)
        
        while len(d):
            # print(f'{outDirectory}[{len(d)}]: {d[-1]}')
            copyfile(f'images/{d.pop()}.png', f'{path}/{i+1}/{len(d)+1}.png')

    docs -= 1