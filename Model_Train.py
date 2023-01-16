import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CoordinateModel import Predictor

running_loss = 0
epoch = 0
model = Predictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

np.load('/home/howl/robosuite/robosuite-1/vision_data_10.npz')
#print(data)
#input_data = data['a']
#target = data['b']
#input_image = data['c']
batch = 32

for i in range(500):
    

    optimizer.zero_grad()
    output = model(input)

    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item() #need to initialise running_loss = 0.0 before the loop
    if batch % 10 == 9:    # print every 10 batches
        print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 10:.8f}')
        running_loss = 0.0

path = 'weights.pt'
torch.save(model.state_dict(), path)
print('Model state dictionary save as: ' + path)  
