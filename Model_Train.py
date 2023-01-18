# Import the necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from CustomGym import RoboEnv
from CoordinateModel import Predictor
from PIL import Image

# Load the data from the npz file
data = np.load("/home/howl/robosuite/robosuite-1/vision_data_10.npz")
image_data = data["images"]
labels = data["labels"]
bounding_boxes = data["data"]

# Preprocess the image data
image_data = image_data.astype(np.uint8)
image_data = np.reshape(image_data, (image_data.shape[0], image_data.shape[1],3))
image_data = image_data / 255.0
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
image_data = [transforms(Image.fromarray(image)) for image in image_data]

# Prepare the data
image_data = torch.stack(image_data)
labels = np.asarray(labels)
bounding_boxes = np.asarray(bounding_boxes)

# Split the data
split = int(0.8 * len(image_data))
train_data = (image_data[:split], labels[:split], bounding_boxes[:split])
test_data = (image_data[split:], labels[split:], bounding_boxes[split:])

# Train the model
model = Predictor()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

for epoch in range(50):
    for data in train_data:
        optimizer.zero_grad()
        output = model(data[0], data[2])
        loss = loss_fn(output, data[1])
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_data:
        output = model(data[0], data[2])
        loss = loss_fn(output, data[1])
        total += data[1].size(0)
        correct += (output == data[1]).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the test data: {}%'.format(accuracy))
