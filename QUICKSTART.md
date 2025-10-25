# 🚀 Quick Start Guide

This guide will help you get started with the Neuropathology Detection System quickly.

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

### Option 1: Run the Web Application (Demo Mode)

The quickest way to see the system in action:

```bash
streamlit run app.py
```

This will:
- Start a web server at http://localhost:8501
- Open your browser automatically
- Show a modern interface for uploading and analyzing MRI images
- Work in demo mode (random predictions for demonstration)

**Note:** Demo mode doesn't require a trained model. For real predictions, you need to train a model first.

### Option 2: Run Validation Script

To check if everything is set up correctly:

```bash
python validate.py
```

This will:
- Check project structure
- Validate Python syntax
- Show project statistics
- Verify documentation

### Option 3: Run Demo Tests

To test all components without a full dataset:

```bash
python demo.py
```

This will:
- Test data loading
- Test model building (all architectures)
- Test predictions
- Test visualizations
- Show a summary of results

**Requires:** All dependencies installed

### Option 4: Train Your Own Model

To train on actual MRI data:

```bash
python train.py --data_dir /path/to/your/dataset --epochs_stage1 30 --epochs_stage2 20
```

**Requirements:**
- A dataset organized in the proper format (see below)
- GPU recommended (CPU training is very slow)

## Dataset Format

Your dataset should be organized like this:

```
your_dataset/
├── Glioma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Meningioma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Pituitary/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Recommended Dataset

**Brain MRI Images for Brain Tumor Detection** (Available on Kaggle)
- ~3000 MRI images
- 4 classes: Glioma, Meningioma, Pituitary, No Tumor
- Pre-processed and ready to use

To use:
1. Download from Kaggle
2. Extract to a directory
3. Run training script with that directory

## What to Explore

### 1. Web Application Features

When you run `streamlit run app.py`, explore:

- **Home Page**: Project overview and key features
- **Detection Page**: Upload images and get predictions
- **About Model Page**: Technical details and architecture
- **Theory Page**: Learn deep learning concepts with detailed explanations

### 2. Code Structure

The codebase is organized for easy learning:

```
src/
├── models/neuropathology_model.py  # Model architecture with theory
├── data/data_loader.py             # Data handling with explanations
└── utils/helpers.py                # Visualization and evaluation
```

Each file contains:
- Comprehensive docstrings
- Mathematical explanations
- Theory correlations
- Code-to-concept mappings

### 3. Training Process

When training a model:

1. **Stage 1: Feature Extraction**
   - Trains only the classification head
   - Takes 20-30 epochs
   - Reaches ~85-90% accuracy

2. **Stage 2: Fine-Tuning**
   - Fine-tunes last layers of base model
   - Takes 10-20 epochs
   - Final accuracy ~95%+

3. **Monitoring**
   - Watch training curves in real-time
   - TensorBoard logs saved to `logs/`
   - Best model saved to `models/`

## Learning Path

This project is designed as a hands-on learning platform:

1. **Start with the Web App**
   - See the system in action
   - Understand the problem domain
   - Explore the UI and features

2. **Read the Theory**
   - Navigate to Theory page in the app
   - Covers neural networks, CNNs, transfer learning
   - Mathematical foundations explained

3. **Explore the Code**
   - Start with `src/models/neuropathology_model.py`
   - Read inline documentation
   - Correlate code with theory

4. **Run Training**
   - Get hands-on with actual training
   - Understand the two-stage process
   - Experiment with hyperparameters

5. **Experiment**
   - Try different architectures
   - Adjust learning rates
   - Implement improvements

## Common Issues

### "Module not found" errors

Make sure you've activated your virtual environment and installed dependencies:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Training is very slow

- **Solution 1**: Use a machine with GPU
- **Solution 2**: Reduce batch size: `--batch_size 16`
- **Solution 3**: Use MobileNet: `--base_model mobilenet`
- **Solution 4**: Reduce epochs: `--epochs_stage1 15 --epochs_stage2 10`

### Out of memory errors

- Reduce batch size: `--batch_size 16` or even `--batch_size 8`
- Use a smaller model: `--base_model mobilenet`
- Close other applications

### Poor accuracy

- Check data quality (corrupted images, wrong labels)
- Increase training time (more epochs)
- Try data augmentation
- Ensure balanced dataset

## Next Steps

1. **Complete the Tutorial**: Read through all theory sections in the app
2. **Train a Model**: Get a dataset and train your first model
3. **Experiment**: Try different architectures and hyperparameters
4. **Extend**: Add new features like Grad-CAM visualizations
5. **Deploy**: Consider deploying to cloud platforms

## Getting Help

- **Documentation**: Read the extensive README.md
- **Code Comments**: All code is thoroughly documented
- **Theory Pages**: In-depth explanations in the web app
- **GitHub Issues**: Report bugs or ask questions

## Important Notes

⚠️ **Medical Disclaimer**: This is an educational tool. Do not use for actual medical diagnosis.

🎓 **Learning Focus**: Take time to understand the theory. The goal is to learn deep learning fundamentals.

💻 **Experimentation**: Don't hesitate to modify the code and try new ideas. That's how you learn!

## Summary of Commands

```bash
# Setup
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt

# Validation
python validate.py

# Demo
python demo.py

# Web App
streamlit run app.py

# Training
python train.py --data_dir /path/to/data --epochs_stage1 30 --epochs_stage2 20
```

Happy learning! 🧠🤖
