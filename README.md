# Knowledge Distillation for 2D Human Pose Estimation

## 📋 Overview

This repository contains a comprehensive **technical session** on applying **Knowledge Distillation (KD)** techniques to 2D Human Pose Estimation. The session demonstrates how to transfer knowledge from a large, accurate Teacher model (HRNet) to a lightweight Student model (SqueezeNet) for efficient deployment.

### What You'll Learn

- **Baseline Training**: Train a lightweight student model from scratch
- **Logits-based KD**: Distill knowledge from teacher's output heatmaps using spatial softmax and temperature scaling
- **Feature-based KD**: Align intermediate layer representations using PyTorch hooks and feature adapters
- **Evaluation**: Assess models using PCK metrics and visualize predictions with FiftyOne

### Key Techniques Covered

1. **Data Preprocessing**: COCO dataset handling, cropping, coordinate transformation, and heatmap generation
2. **Model Architectures**: HRNet (Teacher) and SqueezeNet (Student) for pose estimation
3. **Knowledge Distillation**: 
   - Spatial KL Divergence with Temperature (T=4.0)
   - MSE-based feature matching
   - PyTorch forward hooks for intermediate feature extraction
4. **Modern Training**: AdamW optimizer, Mixed Precision (AMP), Cosine Annealing with Warmup
5. **Complexity Analysis**: FLOPs/MACs calculation using `calflops`

---

## 🚀 Quick Start

### Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mwangi-clinton/2d-human-pose-estimation-using-kd/blob/main/knowledge-distillation-hpe.ipynb)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 20GB+ disk space for COCO dataset

### Step 1: Clone the Repository
```bash
git clone https://github.com/mwangi-clinton/2d-human-pose-estimation-using-kd.git
cd 2d-human-pose-estimation-using-kd
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy tqdm matplotlib
pip install fiftyone calflops
pip install heatmaps-to-keypoints  
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Download COCO Dataset
The notebook includes automated download scripts, but you can manually download:
```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip annotations_trainval2017.zip
unzip val2017.zip
unzip train2017.zip
```

---

## Usage

### Running the Notebook

1. **Launch Jupyter**:
   ```bash
   jupyter notebook knowledge-distillation-hpe.ipynb
   ```

2. **Execute Cells Sequentially**:
   - **Section 1**: Dataset loading and preprocessing
   - **Section 2**: Model definitions (HRNet, SqueezeNet)
   - **Section 3**: Baseline student training
   - **Section 4**: Logits-based KD training
   - **Section 5**: Feature-based KD training
   - **Section 6**: Evaluation and comparison



## Contact

For questions or collaboration opportunities:
- GitHub: [@mwangi-clinton](https://github.com/mwangi-clinton)
- Email: your-email@example.com
