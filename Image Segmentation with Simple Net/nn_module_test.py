import torch
import cv2 as cv
import numpy as np
import os
import glob
import pydicom as dcm
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

testing_images = []
testing_labels = []

def load_images_from_folder(folder, images=None):
    os.chdir(folder)
    print(f"\nProcessing subfolder: {folder}")
    for img_path in glob.glob("*.jpg"):
        img = cv.imread(img_path)
        if img is None:
            print(f"Image load failed for {img_path}")
            continue
            
        # Convert to grayscale
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Normalize the image
        img_gray = cv.normalize(img_gray, None, 0, 1, cv.NORM_MINMAX)
        img_gray = img_gray.astype(np.float32)  # Ensure the image is in float32 format
            
        images.append(img_gray)
    os.chdir("..")

def load_dicom_images_from_folder(folder, images=None) -> bool:
    os.chdir(folder)
    print(f"\nProcessing subfolder: {folder}")
    g = glob.glob("*.dcm")
    slices = [dcm.dcmread(f) for f in g]
    if len(slices) == 0:
        print(f"No DICOM files found in {folder}")
        return False
    largest_pixel_value = slices[0][0x00280107].value # Dicom Dataset
    for idx in range(len(slices)):
        image = slices[idx].pixel_array # Dicom Dataset
        image = cv.convertScaleAbs(image, alpha=(255.0/largest_pixel_value))
        image = cv.normalize(image, None, 0, 10, cv.NORM_MINMAX)
        image = image.astype(np.float32)  # Ensure the image is in float32 format
        images.append(image)
    os.chdir("..")
    return True

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct/len(y_pred)) * 100
    return acc

def sigmoid_to_mask(tensor, threshold=0.5):
    return (tensor > threshold).float()

### Define a Model
class SimpleSegNet(nn.Module):
    def __init__(self):
        super(SimpleSegNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32 -> 64
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 64 -> 128
        self.up3 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)   # 128 -> 256

    def forward(self, x):
        x = torch.relu(self.conv1(x))    # (B, 16, 256, 256)
        x = self.pool1(x)            # (B, 16, 128, 128)
        x = torch.relu(self.conv2(x))    # (B, 32, 128, 128)
        x = self.pool2(x)            # (B, 32, 64, 64)
        x = torch.relu(self.conv3(x))    # (B, 64, 64, 64)
        x = self.pool3(x)            # (B, 64, 32,
        x = torch.relu(self.up1(x))            # (B, 32, 64, 64)
        x = torch.relu(self.up2(x))          # (B, 32, 64,
        x = self.up3(x)          # (B, 1, 256, 256)
        return x

### 0. Set the path to the training data folder
data_path = os.path.join(os.getcwd(), "Data")
if os.path.exists(data_path):
    os.chdir(data_path)
else:
    print("Can not find the Data folder, please check the path.")
    quit()

print("Reading Testing MR Images...")
print(f"{os.getcwd()}\\Testing Set\\*\\MR")
### 1.2 Read Testing MR Images
for subfolder in tqdm(glob.glob(f"{os.getcwd()}\\Testing Set\\*\\MR")):
    if os.path.isdir(subfolder):
        load_dicom_images_from_folder(subfolder, testing_images)
os.chdir(data_path)

print("Reading Testing Mask Maps...")
print(f"{os.getcwd()}\\Testing Set\\*\\Mask")
### 1.3 Read Testing Mask Maps
for subfolder in tqdm(glob.glob(f"{os.getcwd()}\\Testing Set\\*\\Mask")):
    if os.path.isdir(subfolder):
        load_images_from_folder(subfolder, testing_labels)
os.chdir(data_path)

testing_images = np.stack(testing_images, axis=0)
testing_labels = np.stack(testing_labels, axis=0)

testing_images_tensor = torch.from_numpy(testing_images).float().to(device)
testing_labels_tensor = torch.from_numpy(testing_labels).float().to(device)
print(f"Testing images shape: {testing_images_tensor.shape}")
print(f"Testing labels shape: {testing_labels_tensor.shape}")

from pathlib import Path
os.chdir("..")
MODEL_PATH = Path("models")
MODEL_NAME = "model_v1.pth"
MODEL_LOAD_PATH = fr"{os.getcwd()}/{MODEL_PATH}/{MODEL_NAME}"
model_test = SimpleSegNet().to(device)
model_test.load_state_dict(torch.load(MODEL_LOAD_PATH))

# Evaluate loaded model
loss_function = nn.BCEWithLogitsLoss()
model_test.eval()
sampled_image = None
sampled_mask = None
with torch.no_grad():
    running_loss = 0.0
    accuracy = 0.0
    for i in range(len(testing_images_tensor)):
        test_image = testing_images_tensor[i].unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        test_label = testing_labels_tensor[i].unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        
        test_pred_logits = model_test(test_image)
        test_loss = loss_function(test_pred_logits, test_label)
        
        test_pred_mask = sigmoid_to_mask(test_pred_logits)
        test_acc = accuracy_fn(test_pred_mask.view(-1), test_label.view(-1))
        
        sampled_image = test_image.squeeze().cpu().numpy()
        sampled_mask = test_pred_mask
        
        running_loss += test_loss.item()
        accuracy += test_acc

    avg_loss = running_loss / len(testing_images_tensor)
    avg_accuracy = accuracy / len(testing_images_tensor)
print(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")

### Visualize the sampled image and mask
sampled_mask = sampled_mask.squeeze().cpu().numpy()  # Remove batch and channel dimensions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sampled_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sampled_mask, cmap='gray')
plt.title("AI Final Output")
plt.axis('off')
plt.show()