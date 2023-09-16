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
import matplotlib.pyplot as plt
from PIL import Image


# CONSTANTS
batch_size = 64
Nepochs = 3
homeDir = r"C:\Users\Asus\Documents\Coding\Python\Machine Learning\ShapeRecognitionCNN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ### Load files and preprocess...
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


    ### Train and test

    model = Net()
    fit(model,train_data_loader,valid_data_loader,Nepochs,0.005)

    ### Test on a specific image
    
    # Load and preprocess the specific image
    images = [homeDir + r'\images\valid\circle\3.png',
              homeDir + r'\images\valid\square\0.png',
              homeDir + r'\images\valid\star\1.png',
              homeDir + r'\images\valid\triangle\1.png',
              homeDir + r'\testImages\circle.png',
              homeDir + r'\testImages\square.png',]
    for image_path in images:
        print("\nCurrently at image:\n ", image_path)
        image = Image.open(image_path)
        # Perform necessary transformations
        transform = transforms.Compose([transforms.Resize((64, 64))
                                        ,transforms.ToTensor()
                                        ])
        input_tensor = transform(image) 
        if input_tensor.size(0) == 1:  # Check if the image has only 1 channel
            input_tensor = torch.cat([input_tensor] * 3)  # Convert to 3 channels (grayscale to RGB)
        normalize = transforms.Normalize(
                            mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
        input_tensor = normalize(input_tensor)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
        input_batch = input_batch.to('cuda')

        # Set the model to evaluation mode
        model.eval()
        # Make a prediction
        with torch.no_grad():
            output = model(input_batch)
        # Interpret the output (e.g., for classification)
        print("Output:", F.softmax(output, dim=1))
        _, predicted_class = output.max(1)

        # Print the predicted class
        print("Predicted Class:", predicted_class.item())
