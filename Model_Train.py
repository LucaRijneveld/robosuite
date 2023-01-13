import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from CoordinateModel import Predictor

model = Predictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

data = np.load('/home/howl/robosuite/vision_data_10.npz')
input_data = data[0]
target = data[1]
input_image = data[2]

batch_size = 32

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
