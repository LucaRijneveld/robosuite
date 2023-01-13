import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(79360, 512)
        self.fc2 = nn.Linear(512, 64)
        #Add in the 8 additional object detection features: 4 for bounding box limits, for for 1 hot encoded class
        #e.g. for class 1: [xmin, ymin, xmax, ymax, 1, 0, 0, 0]
        #e.g. for class 3: [xmin, ymin, xmax, ymax, 0, 0, 1, 0]
        self.fc3 = nn.Linear(64+8, 32)
        #Final layer with 4 outputs (i.e. [x, y, z, yaw])
        self.fc4 = nn.Linear(32, 4)

    def forward(self, x, x2):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        concx = torch.cat((x,x2),dim=1) #concatenate the additional features
        x = F.relu(self.fc3(concx)) 
        x = self.fc4(x)
        return x

