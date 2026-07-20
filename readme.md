# PERCEPT-Net
**A Perceptual Loss–Driven Framework for Reducing MRI Artifact–Tissue Confusion**

Official PyTorch implementation of PERCEPT-Net, a deep learning framework for MRI artifact reduction. It introduces a self-supervised pre-trained perceptual feature extractor to construct perceptual loss, which effectively alleviates the confusion between imaging artifacts and normal tissue structures, and improves the visual fidelity and structural accuracy of artifact removal results.

---

## Environment & Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.10.0 (CUDA version recommended)
- pydicom >= 2.3.0
- NumPy >= 1.21
- tqdm >= 4.62

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/Radyn759/PERCEPT-Net.git
cd PERCEPT-Net

pip install torch torchvision
pip install pydicom numpy tqdm

# Data Preparation
The framework supports DICOM format MRI series data. You need to prepare paired input (with artifacts) and ground truth (artifact-free) datasets, organized in the following directory structure:

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
- Each subfolder corresponds to one complete MRI series, containing all slice DICOM files of the series.
- The series IDs of input data and ground truth must be exactly the same for one-to-one correspondence.
- The framework will automatically read and sort slices by SliceLocation / InstanceNumber, and perform standard DICOM grayscale correction and percentile truncation normalization.

For perceptual encoder pre-training, it is recommended to use artifact-free ground truth data, so that the encoder can learn more accurate normal tissue feature representation.

# Usage Guide

