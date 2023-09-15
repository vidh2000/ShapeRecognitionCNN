import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from model import *
import os 

# CONSTANTS
batch_size = 64
Nepochs = 20
homeDir = r"C:\Users\Asus\Documents\Coding\Python\Machine Learning\ShapeRecognitionCNN"

if __name__ == "__main__":

    # Load files and preprocess...
    simple_transform = transforms.Compose([transforms.Resize((64, 64))
                                        ,transforms.ToTensor()
                                        ,transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                        ])

    train = ImageFolder(os.path.join(homeDir,'images/train'),simple_transform)
    valid = ImageFolder(os.path.join(homeDir,'images/valid'),simple_transform)

    train_data_loader = torch.utils.data.DataLoader(train,
                                        batch_size=batch_size,
                                        num_workers=3,
                                        shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid,
                                        batch_size=batch_size,
                                        num_workers=3,
                                        shuffle=True)


    # Train and test

    model = Net()
    fit(model,train_data_loader,valid_data_loader,Nepochs,0.01)







