# Knowledge Distillation for 2D Human Pose Estimation

##  Overview

This repository contains a **technical session** on applying **Knowledge Distillation (KD)** techniques to 2D Human Pose Estimation. The session demonstrates how to transfer knowledge from a large, accurate Teacher model (HRNet) to a lightweight Student model (SqueezeNet) for efficient deployment.

### Topics covered

- **Baseline Training**: Train a lightweight student model from scratch
- **Logits-based KD**: Distill knowledge from teacher's output heatmaps using spatial softmax and temperature scaling
- **Evaluation**: Assess models using PCK metrics and visualize predictions with FiftyOne

---

## Quick Start

### Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mwangi-clinton/2d-human-pose-estimation-using-kd/blob/main/kd-hpe.ipynb)

---

## Installation

### Prerequisites
- Python 3.8+
- 20GB+ disk space for COCO dataset

### Step 1: Clone the Repository
```bash
git clone https://github.com/mwangi-clinton/2026-02-02-knowledge-distillation-for-2d-hpe.git
cd 2026-02-02-knowledge-distillation-for-2d-hpe
```

### Step 2: Create Virtual Environment
It is recommended to create a virtual environment to isolate dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
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

### Step 4: Download COCO Dataset
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
   jupyter notebook knowledge-distillation-for-2d-hpe.ipynb
   ```

