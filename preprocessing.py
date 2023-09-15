import numpy as np 
import os 
from math import floor
from random import shuffle
import shutil

home = os.getcwd()
dataDir = r"C:\Users\Asus\Documents\Coding\Python\Machine Learning\ShapeRecognitionCNN\data"

def makeFolders():
    exists = True

    fdr = dataDir + "/train"
    if not os.path.exists(fdr):
        os.mkdir(os.path.join(dataDir, 'train'))
        exists=False
    fdr = dataDir + "/valid"
    if not os.path.exists(fdr):
        os.mkdir(os.path.join(dataDir, 'valid'))
        exists=False

    if not exists:
        for t in ['train', 'valid']:
            for folder in ['circle/', 'square/', 'star/', 'triangle/']:
                os.mkdir(os.path.join(dataDir, t, folder))
    print("Do folders already exist?", exists)

def getTrainTestSplit(dirName, ext):
    allFiles = list()
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if file.endswith(ext):
                allFiles.append(os.path.join(root, file))
    shuffle(allFiles)
    split=0.7
    split_index = floor(len(allFiles) * split)
    training = allFiles[:split_index]
    testing = allFiles[split_index:]
    return training, testing





############################################################
### Make train/valid folders
print("### Make train/valid folders\n")
makeFolders()

############################################################
### Sort train/valid shuffle 70/30 split    
print("\n### Sort train/valid shuffle 70/30 split\n")

rawDataDir = os.path.join(home, 'rawdata')
dataDir = dataDir

train, test = getTrainTestSplit(rawDataDir,"png") #= train, test
print(len(test),len(train))

# Put train/test pics into right folders
for i, filename in enumerate(train):
    filenameNew = filename.replace("rawdata\shapes", r"data\train")
    shutil.copy(filename,filenameNew)


for i, filename in enumerate(test):
    filenameNew = filename.replace("rawdata\shapes", r"data\valid")
    shutil.copy(filename,filenameNew)
 