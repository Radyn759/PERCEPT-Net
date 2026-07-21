# PERCEPT-Net

**A Perceptual Loss–Driven Framework for Reducing MRI Artifact–Tissue Confusion**

Official PyTorch implementation of **PERCEPT-Net**, a deep learning framework for MRI artifact reduction. PERCEPT-Net introduces a self-supervised pre-trained perceptual feature encoder to construct perceptual loss, effectively alleviating the confusion between imaging artifacts and normal tissue structures while improving both visual fidelity and structural accuracy of restored MRI images.

---

# Environment & Installation

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0 (CUDA version recommended)
- torchvision >= 0.11.0
- pydicom >= 2.3.0
- NumPy >= 1.21
- tqdm >= 4.62

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Radyn759/PERCEPT-Net.git
cd PERCEPT-Net

pip install torch torchvision
pip install pydicom numpy tqdm
```

---

# Data Preparation

The framework supports paired **DICOM MRI series**.

The dataset should be organized as follows:

```text
data_root/
├── input/                    # Artifact-corrupted MRI
│   ├── series_001/
│   │   ├── IM-0001-0001.dcm
│   │   ├── IM-0001-0002.dcm
│   │   └── ...
│   ├── series_002/
│   └── ...
└── gt/                       # Artifact-free ground truth
    ├── series_001/
    ├── series_002/
    └── ...
```

Each subdirectory corresponds to one complete MRI series containing all DICOM slices.

## Notes

- The series names under **input** and **gt** must be identical for one-to-one correspondence.
- DICOM slices are automatically sorted using:
  - `SliceLocation`
  - `InstanceNumber`
- Standard DICOM grayscale correction and percentile truncation normalization are performed automatically.
- For perceptual encoder pre-training, artifact-free ground truth images are recommended so that the encoder can learn accurate tissue representations.

---

# Usage

## 1. Pre-train the Perceptual Feature Encoder

The perceptual encoder is first trained via a self-supervised image reconstruction task.

```bash
python train_perc.py \
    --input_dir ./data/gt \
    --output_dir ./percept_ckpts \
    --epochs 50 \
    --batch_size 2 \
    --lr 2e-4 \
    --device cuda
```

### Important Arguments

| Argument | Description |
|----------|-------------|
| `--input_dir` | Ground-truth MRI directory (recommended) |
| `--output_dir` | Directory for saving encoder checkpoints |
| `--val_split` | Validation ratio (default: 0.15) |
| `--base_dim` | Base feature dimension (default: 32) |
| `--use_amp` | Enable mixed precision training (default: True) |

After training, the best encoder checkpoint will be saved as:

```text
best_percept_encoder.pth
```

This checkpoint is required during the main training stage.

---

## 2. Main Network Training

Train the MRI artifact removal model with joint supervision of:

- GAN loss
- Pixel loss
- Perceptual loss

```bash
python train.py \
    --input_dir ./data/input \
    --gt_dir ./data/gt \
    --output_dir ./checkpoints \
    --percept_ckpt ./percept_ckpts/best_percept_encoder.pth \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --lambda_gan 1.0 \
    --lambda_pixel 1.0 \
    --lambda_percept 1.0 \
    --device cuda
```

### Important Arguments

| Argument | Description |
|----------|-------------|
| `--percept_ckpt` | Path to the pre-trained perceptual encoder |
| `--lambda_gan` | Weight of GAN loss |
| `--lambda_pixel` | Weight of pixel loss |
| `--lambda_percept` | Weight of perceptual loss |
| `--nc` | UNet encoder channels (default: `[64,128,256,512]`) |
| `--nb` | Number of residual blocks per stage (default: `4`) |

Training checkpoints are automatically saved every **10 epochs**, including:

- Generator
- Discriminator
- Optimizer
- Learning-rate scheduler

This supports training resumption from checkpoints.

---

## 3. Inference

Run MRI artifact removal using a trained generator.

```bash
python inference.py \
    --input_dir ./data/test_input \
    --output_dir ./results \
    --model_path ./checkpoints/G_epoch100.pth \
    --device cuda
```

### Important Arguments

| Argument | Description |
|----------|-------------|
| `--input_dir` | Input DICOM directory |
| `--output_dir` | Output directory |
| `--model_path` | Trained generator checkpoint |
| `--device` | `cuda` or `cpu` |

---

# Repository Structure

```text
PERCEPT-Net/
├── perceptnet/
│   └── network_unet.py      # UNet / UNetRes generator
├── utils/                   # Utility scripts
├── input_desensitized/      # Example desensitized MRI data
├── train.py                 # Main GAN training
├── train_perc.py            # Perceptual encoder pre-training
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
└── README.md
```

---

# Core Design

## Self-supervised Perceptual Encoder

The perceptual encoder is pre-trained through an image reconstruction task, enabling it to learn multi-scale structural representations of normal tissues without manual annotations. The extracted features provide semantic supervision during artifact removal.

---

## Multi-scale Perceptual Loss

Perceptual loss is computed by measuring the L1 distance between encoder features at multiple scales.

This encourages:

- structural consistency,
- preservation of anatomical details,
- reduction of tissue blurring,
- suppression of pseudo-structure generation.

---

## Residual UNet Generator

The generator adopts a residual UNet architecture with global residual learning, allowing the network to focus on predicting artifact components while preserving normal anatomical structures.

---

## PatchGAN Discriminator

A PatchGAN discriminator performs local patch-wise discrimination, improving the recovery of fine textures and high-frequency image details.

---

# Training Recommendations

- **Always use the pre-trained perceptual encoder** during the main training stage. Randomly initialized feature extractors cannot provide meaningful semantic constraints.
- The default training strategy is **series-level sampling** with `batch_size=1`.
- For high-resolution MRI data, reducing the channel configuration (`nc`) or enabling mixed precision training is recommended to reduce GPU memory usage.

---

# Data Consistency

The preprocessing pipelines used for:

- perceptual encoder pre-training
- main network training

must remain **identical**.

Changing normalization or grayscale processing for only one stage will invalidate the perceptual loss and degrade training performance.

---

# License

This project is provided **for academic research purposes only**.

Commercial use is currently prohibited.

All rights are reserved by the authors.

# TODO / Roadmap

The following components are currently under active development to improve reproducibility and usability.

## Data Processing Pipeline

- [ ] Provide a complete DICOM-to-training-data conversion pipeline.

  The planned pipeline will include:

  - DICOM series loading
  - Slice ordering based on spatial information
  - Intensity correction and normalization
  - Paired dataset generation
  - Export to training-ready formats (`.npy` / `.h5` / `.pt`)

  Example workflow:

  ```text
  Raw DICOM Series
          |
          ↓
  DICOM preprocessing
          |
          ↓
  Intensity normalization
          |
          ↓
  Training dataset generation
          |
          ↓
  PERCEPT-Net training
