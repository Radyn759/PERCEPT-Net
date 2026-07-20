PERCEPT-Net
A Perceptual Loss–Driven Framework for Reducing MRI Artifact–Tissue Confusion
Official PyTorch implementation of PERCEPT-Net, a deep learning framework for MRI artifact reduction. It introduces a self-supervised pre-trained perceptual feature extractor to construct perceptual loss, which effectively alleviates the confusion between imaging artifacts and normal tissue structures, and improves the visual fidelity and structural accuracy of artifact removal results.
Environment & Installation
Requirements
Python >= 3.8
PyTorch >= 1.10.0 (CUDA version recommended)
pydicom >= 2.3.0
NumPy >= 1.21
tqdm >= 4.62
Installation Steps
Clone the repository:
bash
运行
git clone https://github.com/Radyn759/PERCEPT-Net.git
cd PERCEPT-Net
Install dependencies:
bash
运行
pip install torch torchvision
pip install pydicom numpy tqdm
Data Preparation
The framework supports DICOM format MRI series data. You need to prepare paired input (with artifacts) and ground truth (artifact-free) datasets, organized in the following directory structure:
plaintext
data_root/
├── input/          # Artifact-corrupted DICOM data
│   ├── series_001/
│   │   ├── IM-0001-0001.dcm
│   │   ├── IM-0001-0002.dcm
│   │   └── ...
│   ├── series_002/
│   └── ...
└── gt/             # Artifact-free ground truth DICOM data
    ├── series_001/
    ├── series_002/
    └── ...
Each subfolder corresponds to one complete MRI series, containing all slice DICOM files of the series.
The series IDs of input data and ground truth must be exactly the same for one-to-one correspondence.
The framework will automatically read and sort slices by SliceLocation / InstanceNumber, and perform standard DICOM grayscale correction and percentile truncation normalization.
For perceptual encoder pre-training, it is recommended to use artifact-free ground truth data, so that the encoder can learn more accurate normal tissue feature representation.
Usage Guide
1. Pre-train Perceptual Feature Encoder
First, pre-train the perceptual feature extractor through an autoencoder reconstruction task, which is the core premise for the perceptual loss to play a semantic constraint role.
bash
运行
python train_perc.py \
  --input_dir ./data/gt \
  --output_dir ./percept_ckpts \
  --epochs 50 \
  --batch_size 2 \
  --lr 2e-4 \
  --device cuda
Key parameters:
--input_dir: Root directory of pre-training data (ground truth data recommended)
--output_dir: Save path for pre-trained encoder weights
--val_split: Validation set ratio, default 0.15
--base_dim: Base channel number of encoder, default 32
--use_amp: Enable mixed precision training, enabled by default
After training, the best encoder weight best_percept_encoder.pth will be saved in the output directory for subsequent main training.
2. Main Model Training
Train the artifact removal generation network with the joint supervision of GAN loss, pixel loss and perceptual loss.
bash
运行
python train.py \
  --input_dir ./data/input \
  --gt_dir ./data/gt \
  --output_dir ./checkpoints \
  --percept_ckpt ./percept_ckpts/best_percept_encoder.pth \
  --epochs 100 \
  --batch_size 1 \
  --lr 1e-4 \
  --lambda_gan 1.0 \
  --lambda_pixel 100.0 \
  --lambda_percept 0.5 \
  --device cuda
Key parameters:
--percept_ckpt: Path of pre-trained perceptual encoder weight (core parameter, random initialization will be used if not specified)
--lambda_gan / lambda_pixel / lambda_percept: Weights of three loss terms
--nc: Channel configuration of UNet encoder, default [64, 128, 256, 512]
--nb: Number of residual blocks in each stage, default 4
Checkpoints will be saved every 10 epochs, including complete training states such as generator, discriminator, optimizer and learning rate scheduler, supporting breakpoint resumption.
3. Inference
Use the trained model to perform artifact removal inference on DICOM data:
bash
运行
python inference.py \
  --input_dir ./data/test_input \
  --output_dir ./results \
  --model_path ./checkpoints/G_epoch100.pth \
  --device cuda
Repository Structure
plaintext
PERCEPT-Net/
├── perceptnet/          # Core network implementation
│   └── network_unet.py  # UNet / UNetRes generator architecture
├── utils/               # Utility scripts
├── input_desensitized/  # Desensitized sample input data
├── train.py             # Main training script (GAN + pixel + perceptual loss)
├── train_perc.py        # Perceptual encoder pre-training script (autoencoder)
├── inference.py         # Inference script
├── requirement          # Dependency list
└── README.md
Core Design
Self-supervised Perceptual Encoder
Pre-trained via image reconstruction task, extracts multi-scale structural features of tissues without manual annotation, and provides semantic-level supervision for artifact removal.
Multi-scale Perceptual Loss
Calculates L1 distance of features at three different scales, constrains high-level structural consistency of images, and reduces the problem of tissue structure blurring or pseudo-structure introduction caused by pixel-level loss.
Residual UNet Generator
Based on the global residual learning design, it directly learns the artifact distribution, with fast convergence and stable training.
PatchGAN Discriminator
Performs local patch-level true-false discrimination, which is conducive to preserving high-frequency details of images.
Notes
Usage License
The model and code are only available for academic research use. Commercial use is prohibited for the time being.
Training Recommendations
Always use the pre-trained perceptual encoder weight for main training. Random initialized feature extractor cannot provide effective semantic constraint.
The default dataset is series-level. It is recommended to set batch_size=1. If you need a larger batch size, you can adjust it to slice-level sampling.
For high-resolution MRI data, you can appropriately reduce the nc channel configuration or enable mixed precision training to save video memory.
Data Consistency
The preprocessing logic of perceptual pre-training and main training is completely aligned. Please do not modify the normalization or grayscale processing of one side alone, otherwise the perceptual loss will be invalid.
License
This project is for research purposes only. All rights reserved by the authors.
