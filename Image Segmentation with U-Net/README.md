# 🧠 PyTorch Image Segmentation on MR Images

This folder provides a simple yet effective pipeline for **image segmentation using PyTorch**, specifically targeting **MR medical images** with corresponding mask annotations. The model is trained to predict segmentation masks from grayscale medical scans using a custom lightweight convolutional neural network.

---

## 📌 Features

- ✅ Supports **DICOM (`.dcm`)** and **JPG mask** image loading
- ✅ Simple encoder-decoder convolutional segmentation network
- ✅ Fully written in **PyTorch**, optimized for GPU (CUDA)
- ✅ Visualizes the final prediction result as a binary mask
- ✅ Supports saving the model state (`.pth`) for later use


---

## 🧪 Dataset Requirements

- Input: DICOM images (`.dcm`) stored under `Training Set/<case>/MR/`
- Label: Corresponding segmentation masks in `.jpg` format under `Training Set/<case>/Mask/`

The script normalizes and resizes all images and labels automatically and converts them into PyTorch tensors.

---

## 🚀 How to Run

#### 1. Clone the repo
```bash
git clone https://github.com/raind-dev/PyTorch.git
cd Image Segmentation with U-Net
```
#### 2. Prepare dataset
Place your MR images and mask maps under the Data/Training Set/ folder following the structure above.

#### 3. Install dependencies
```bash
pip install torch torchvision opencv-python pydicom matplotlib tqdm
```

#### 4. Run training
```bash
python nn_module.py
```

#### 5. Run testing
```bash
python nn_module_test.py
```

## 🧠 Model Architecture
The model is a custom SimpleSegNet, consisting of:

Encoder: L1 ~ L4 U-Net architecture. (two convolution layers and two LeakyRelu layers per dimension)

Decoder: L1 ~ L4 U-Net architecture.

Activation: LeakyReLU throughout, final layer without activation (use with BCEWithLogitsLoss)

## 🎯 Training Details
Loss Function: BCEWithLogitsLoss

Optimizer: Adam with learning rate 0.01

Epochs: 10

Input Shape: (1, 256, 256)

Evaluation Metric: Pixel-wise accuracy

## 📊 Visualization Output
At the end of training, the script shows:

Original grayscale input image

Predicted binary segmentation mask

## 💾 Model Saving & Loading
The trained model is saved as a .pth file:
```python
# Save model
torch.save(model.state_dict(), "models/model_v1.pth")

# Load model
model = SegUNetL1() ~ SegUNetL4()
model.load_state_dict(torch.load("models/model_v1.pth"))
```



