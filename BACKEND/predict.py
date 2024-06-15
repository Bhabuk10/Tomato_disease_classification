import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


# Define the CNN architecture
# class PlantVillageCNN(nn.Module):
#     def __init__(self):
#         super(PlantVillageCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(8 * 16 * 16, 128)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.pool1(nn.functional.relu(self.conv1(x)))
#         x = self.pool2(nn.functional.relu(self.conv2(x)))
#         x = self.pool3(nn.functional.relu(self.conv3(x)))
#         x = x.view(-1, 8 * 16 * 16)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x



class ImprovedPlantVillageCNN(nn.Module):
    def __init__(self):
        super(ImprovedPlantVillageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# Load the trained PyTorch model
model = ImprovedPlantVillageCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(r'C:\Users\User\Desktop\Tomato_disease_classification\ML\model_1.pth'))
# model.load_state_dict(torch.load(r'C:\Users\User\Desktop\Tomato_disease_classification\ML\model_1.pth'))
# model.load_state_dict(torch.load('/app/ML/model_1.pth'))

model.eval()

# Define the class labels
class_labels = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
                'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# # Define the image preprocessing transforms
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]),
# }


# def predict_image(image_data):
#     # Load and preprocess the image data
#     img = Image.open(image_data)
#     img = preprocess(img)
#     img = img.unsqueeze(0)  # Add batch dimension

#     # Make the prediction
#     with torch.no_grad():
#         outputs = model(img)
#         _, predicted = torch.max(outputs.data, 1)

#     # Get the predicted class label
#     predicted_class = class_labels[predicted.item()]

#     return predicted_class

from torchvision import transforms
from PIL import Image
import torch

# Define the image preprocessing transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
}

def predict_image(image_data, phase='val'):
    # Ensure phase is either 'train' or 'val'
    if phase not in data_transforms:
        raise ValueError("Phase should be 'train' or 'val'")
    
    # Load and preprocess the image data
    img = Image.open(image_data)
    img = data_transforms[phase](img)  # Apply the correct transform
    img = img.unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)

    # Get the predicted class label
    predicted_class = class_labels[predicted.item()]

    return predicted_class
