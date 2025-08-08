import torch
import cv2 as cv
import numpy as np
import os
import glob
import pydicom as dcm
import matplotlib.pyplot as plt
import U_Net_classes as unet

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
    probs = torch.sigmoid(tensor)
    return (probs > threshold).float()

### 0. Set the path to the Testing data folder
data_path = os.path.join(os.getcwd(), "Data")
if os.path.exists(data_path):
    os.chdir(data_path)
else:
    print("Can not find the Data folder, please check the path.")
    quit()

### 1. Read Testing MR Images
print("Reading Testing MR Images...")
print(f"{os.getcwd()}\\Testing Set\\*\\MR")
for subfolder in tqdm(glob.glob(f"{os.getcwd()}\\Testing Set\\*\\MR")):
    if os.path.isdir(subfolder):
        load_dicom_images_from_folder(subfolder, testing_images)
os.chdir(data_path)

print("Reading Testing Mask Maps...")
print(f"{os.getcwd()}\\Testing Set\\*\\Mask")
### 1.1 Read Testing Mask Maps
for subfolder in tqdm(glob.glob(f"{os.getcwd()}\\Testing Set\\*\\Mask")):
    if os.path.isdir(subfolder):
        load_images_from_folder(subfolder, testing_labels)
os.chdir(data_path)
         
### 2. Convert to Tensor
testing_images = np.stack(testing_images, axis=0)
testing_labels = np.stack(testing_labels, axis=0)

testing_images_tensor = torch.from_numpy(testing_images).float().to(device)
testing_labels_tensor = torch.from_numpy(testing_labels).float().to(device)

print(f"Testing images shape: {testing_images_tensor.shape}")
print(f"Testing labels shape: {testing_labels_tensor.shape}")

conv = nn.Conv2d(1, 16, kernel_size=3, padding=1).to(device)
output = conv(testing_images_tensor.unsqueeze(1).float())

print(f"Output shape after convolution: {output.shape}")

### 4. Create an instance of the model, loss function, and optimizer
from pathlib import Path
os.chdir("..")
MODEL_PATH = Path("models")
MODEL_NAME = "model_v1.pth"
MODEL_LOAD_PATH = fr"{os.getcwd()}/{MODEL_PATH}/{MODEL_NAME}"
model_test = unet.SegUNet_L1().to(device)
model_test.load_state_dict(torch.load(MODEL_LOAD_PATH))

loss_function = nn.BCEWithLogitsLoss() 

### 5. Training Loop
epochs = 100
final_output = None
final_image = None
model_test.eval()
with torch.no_grad():
    running_loss = 0.0
    accuracy = 0.0
    for i in range(len(testing_images_tensor)):
        inputs = testing_images_tensor[i].unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        labels = testing_labels_tensor[i].unsqueeze(0).unsqueeze(0).float()  # Ensure labels are float type
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model_test(inputs)
        final_image = inputs.squeeze().cpu().numpy()  # Save the input image for visualization
        final_output = outputs
        #print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
        
        loss = loss_function(outputs, labels)
        mask_map = sigmoid_to_mask(outputs)
        accuracy += accuracy_fn(mask_map.view(-1), labels.view(-1))

    avg_loss = running_loss / len(testing_images_tensor)
    avg_accuracy = accuracy / len(testing_images_tensor)
print(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%")

### 6. Visualization of the final output
# Final output is a tensor of shape (1, 1, 256, 256) with values between 0 and 1.
print(f"Final output shape: {final_output.shape}")
print(f"Final output values range: {final_output.min().item()} to {final_output.max().item()}")

pred_mask = (torch.sigmoid(final_output) > 0.5).squeeze().cpu().numpy()  # Convert to mask
print(f"Probability map shape: {pred_mask.shape}")
print(f"Probability map values range: {pred_mask.min().item()} to {pred_mask.max().item()}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(final_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap='gray')
plt.title("AI Final Output")
plt.axis('off')
plt.show()
