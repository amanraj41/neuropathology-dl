# 🚀 Quick Start Guide

Get started with the Neuropathology Detection System in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning framework)
- Streamlit (web application)
- NumPy, Pandas (data manipulation)
- Matplotlib, Seaborn, Plotly (visualization)
- Scikit-learn (machine learning utilities)
- OpenCV (image processing)

## Usage Options

### Option 1: Run the Web Application

The quickest way to see the system in action:

```bash
streamlit run app.py
```

This will:
- Start a web server at http://localhost:8501
- Open your browser automatically
- Show a modern interface for uploading and analyzing MRI images
- Load pre-trained models with 78.13% test accuracy

**Features:**
- Model selection dropdown
- Image upload or URL fetch
- Grad-CAM anomaly localization with adjustable sensitivity
- Real-time predictions with confidence scores
- Clinical descriptions for all 17 classes

### Option 2: Train Your Own Model

To train on actual MRI data:

```bash
python train.py --data_dir ./data/brain_mri_17 --base_model mobilenet --epochs_stage1 50 --epochs_stage2 70
```

**Requirements:**
- A dataset organized with one folder per class
- GPU recommended (CPU training is slow)

## Dataset Format

Your dataset should be organized with one folder per class:

```
your_dataset/
├── Class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class_2/
│   ├── image1.jpg
│   └── ...
└── Class_N/
    └── ...
```

### Recommended Dataset

**Kaggle Brain MRI 17-Class Dataset**
- 17 neuropathology classes across T1, T1C+, T2 modalities
- Pre-processed and ready to use
- Comprehensive clinical spectrum

To use:
1. Download from Kaggle
2. Extract to `data/brain_mri_17/`
3. Run training: `python train.py --data_dir data/brain_mri_17`

## What to Explore

### 1. Web Application Features

When you run `streamlit run app.py`, explore:

- **Home Page**: Project overview and key features
- **Diagnostic Classes Page**: All 17 neuropathology categories with MRI modality explanations
- **Detection Page**: Load models, upload images, get predictions with Grad-CAM overlays
- **About Model Page**: Architecture details and performance metrics

### 2. Grad-CAM Explainability

The app includes visual explanations for model decisions:

- **Heatmap Overlay**: Red/yellow areas show suspicious regions, blue shows normal
- **Green Contours**: Boundaries of detected anomalous regions
- **Adjustable Sliders**:
  - Detection Sensitivity (0.1-0.9): Controls which regions are highlighted
  - Overlay Intensity (0.1-0.8): Controls transparency of heatmap
- **Real-time Updates**: Changes update instantly without re-analysis

### 3. Training Process

When training a model:

1. **Stage 1: Feature Extraction**
   - Trains only the classification head
   - Takes 50 epochs
   - Reaches ~85-90% accuracy

2. **Stage 2: Fine-Tuning**
   - Fine-tunes last 20 layers of base model
   - Takes 70 epochs
   - Final accuracy ~78%+

3. **Monitoring**
   - Watch training curves in real-time
   - TensorBoard logs saved to `logs/`
   - Best model saved to `models/`

## Common Issues

### "Module not found" errors

Make sure you've activated your virtual environment and installed dependencies:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Training is very slow

- **Solution 1**: Use a machine with GPU
- **Solution 2**: Reduce batch size: `--batch_size 16` or `--batch_size 8`
- **Solution 3**: Reduce epochs: `--epochs_stage1 30 --epochs_stage2 20`

### Out of memory errors

- Reduce batch size: `--batch_size 8`
- Close other applications

### Poor accuracy

- Check data quality (corrupted images, wrong labels)
- Increase training time (more epochs)
- Ensure balanced dataset

## Next Steps

1. **Explore the App**: Run `streamlit run app.py` and test with different MRI images
2. **Train a Model**: Get the dataset and train your first model
3. **Experiment**: Try different threshold values for Grad-CAM
4. **Extend**: Add new features or improve the model
5. **Deploy**: Consider deploying to cloud platforms

## Summary of Commands

```bash
# Setup
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt

# Web App
streamlit run app.py

# Training
python train.py --data_dir /path/to/data --epochs_stage1 50 --epochs_stage2 70
```

## Important Notes

⚠️ **Medical Disclaimer**: This is a demonstration tool. Do not use for actual medical diagnosis.

💻 **Experimentation**: Modify the code and try new ideas!

📊 **Grad-CAM**: Adjust sliders to see how detection sensitivity affects visualization.

---

**Happy analyzing! 🧠🤖 | Version 1.1**
