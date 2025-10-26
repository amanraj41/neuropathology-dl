# 🧠 Neuropathology Detection System

A comprehensive deep learning system for detecting neuropathological conditions from brain MRI scans. This project combines state-of-the-art computer vision techniques with an intuitive web interface, designed as a practical reference implementation for medical image classification.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete neuropathology detection system powered by deep learning for medical image (MRI) analysis. It features:

- **Transfer Learning**: Leverages pre-trained MobileNetV2 model trained on ImageNet
- **Fine-tuning**: Adapts general image features to medical imaging domain
- **Modern Web Interface**: Interactive Streamlit application for real-time predictions


### Detected Conditions

The system can classify brain MRI scans into **17 distinct categories**, providing comprehensive neuropathological analysis across multiple imaging modalities (T1, T1C+, T2):

**Primary Tumor Categories:**
1-3. **Glioma** (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) - T1, T1C+, T2
4-6. **Meningioma** (de Baixo Grau, Atípico, Anaplásico, Transicional) - T1, T1C+, T2
7-8. **NORMAL** - T1, T2
9-11. **Neurocitoma** (Central - Intraventricular, Extraventricular) - T1, T1C+, T2
12-14. **Outros Tipos de Lesões** (Abscessos, Cistos, Encefalopatias Diversas) - T1, T1C+, T2
15-17. **Schwannoma** (Acustico, Vestibular - Trigeminal) - T1, T1C+, T2

Each class includes detailed clinical information and characteristic MRI findings.

## ✨ Features

### Technical Features

- **Transfer Learning with MobileNetV2**: Efficient pre-trained base model (ImageNet weights)
- **Two-Stage Training**: Feature extraction followed by fine-tuning for optimal results
- **Data Augmentation**: Rotation, zoom, and flip transformations to improve generalization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Model Callbacks**: Early stopping, learning rate scheduling, model checkpoints
- **Visualization Tools**: Training curves, confusion matrices, prediction confidence
- **Automatic Fallback**: Robust model loading with fallback mechanisms

### User Interface Features

- **Modern Web Design**: Clean, intuitive Streamlit interface
- **Manual Model Loading**: Load trained models on-demand with dropdown selection
- **Real-time Predictions**: Upload MRI images or fetch from URL for instant diagnosis
- **Confidence Scores**: Detailed probability distributions for all 17 classes
- **Clinical Information**: Comprehensive medical descriptions and MRI findings for each diagnosed pathology
- **Interactive Visualizations**: Plotly charts for prediction analysis
- **Responsive Layout**: Works on desktop and mobile devices

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                      │
│                    (Streamlit Web App)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Image Upload │  │  Prediction  │  │Visualization │      │
│  │   Handler    │  │    Engine    │  │   Module     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Deep Learning Layer                          │
│  ┌──────────────────────────────────────────────────┐        │
│  │           Pre-trained Base Model                 │        │
│  │              (MobileNetV2)                       │        │
│  └──────────────────────┬───────────────────────────┘        │
│  ┌──────────────────────▼───────────────────────────┐        │
│  │         Custom Classification Head               │        │
│  │  (Dense Layers + Dropout + Batch Norm)           │        │
│  └──────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Data Processing Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Image Loading │  │Preprocessing │  │ Augmentation │      │
│  │              │  │ & Resizing   │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit installed successfully')"
```

## 📖 Usage

### Running the Web Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Training a Model

To train a model on your own dataset:

```bash
python train.py --data_dir /path/to/your/dataset --epochs_stage1 30 --epochs_stage2 20
```

Notes:
- The dataset must be organized as one folder per class (subfolder names become class labels).
- During training, the pipeline saves `models/class_names.json` and `models/metrics.json` so the app can display the correct labels and accuracy for any dataset size (e.g., 17 classes).

#### Training Arguments

```bash
python train.py \
    --data_dir ./data/brain_mri_data \  # Path to dataset
    --base_model mobilenet \            # Model architecture (mobilenet recommended)
    --batch_size 16 \                   # Batch size (16 for 4-core CPU)
    --epochs_stage1 10 \                # Feature extraction epochs
    --epochs_stage2 5 \                 # Fine-tuning epochs
    --learning_rate 0.001 \             # Initial learning rate
    --learning_rate_finetune 0.0001 \   # Fine-tuning learning rate
    --trainable_layers 20               # Layers to fine-tune
```

### Using a different dataset (e.g., 17 classes)

1) Download and extract your dataset so it looks like:

```
data/your_dataset/
├── ClassA/
├── ClassB/
└── ... (N class folders)
```

2) Train:

```bash
python train.py --data_dir data/your_dataset --base_model mobilenet --batch_size 16 --epochs_stage1 10 --epochs_stage2 5
```

3) Run the app:

```bash
streamlit run app.py
```

The app will auto-load `models/final_model.keras`, class names from `models/class_names.json`, and display metrics from `models/metrics.json` if present.

### Making Predictions

#### Using Python Script

```python
from src.models.neuropathology_model import NeuropathologyModel
from src.data.data_loader import MRIDataLoader
import numpy as np

# Load model
model = NeuropathologyModel()
model.load_model('models/final_model.keras')

# Load and preprocess image
loader = MRIDataLoader(img_size=(224, 224))
image = loader.load_and_preprocess_image('path/to/mri_scan.jpg')
image_batch = np.expand_dims(image, axis=0)

# Make prediction
predictions = model.predict(image_batch)
predicted_class, confidence = model.predict_class(image_batch)

print(f"Predicted: {predicted_class}, Confidence: {confidence}")
```

## 📊 Dataset

### Expected Dataset Structure

```
data/brain_mri_17/
├── Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1/
├── Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+/
├── Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2/
├── Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1/
├── Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+/
├── Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2/
├── NORMAL T1/
├── NORMAL T2/
├── Neurocitoma (Central - Intraventricular, Extraventricular) T1/
├── Neurocitoma (Central - Intraventricular, Extraventricular) T1C+/
├── Neurocitoma (Central - Intraventricular, Extraventricular) T2/
├── Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1/
├── Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+/
├── Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2/
├── Schwannoma (Acustico, Vestibular - Trigeminal) T1/
├── Schwannoma (Acustico, Vestibular - Trigeminal) T1C+/
└── Schwannoma (Acustico, Vestibular - Trigeminal) T2/
```

### Recommended Dataset

**Kaggle Brain MRI 17-Class Dataset** (used for this implementation)
- **17 neuropathology classes** across multiple MRI modalities
- **Comprehensive clinical spectrum**: Primary tumors, benign lesions, normal scans, and diverse pathologies
- **Multiple imaging modalities**: T1, T1 post-contrast (T1C+), T2-weighted sequences
- **Pre-processed and ready to use**
- **Source**: [Kaggle - Brain MRI Classification](https://www.kaggle.com/)

**Note**: This system's clinical descriptions and MRI findings are specifically tailored for the 17-class dataset. Using a different dataset will require updating the class descriptions in `src/utils/helpers.py` to match your specific pathology classes.

### Data Preprocessing

The system automatically handles:
- Resizing to 224×224 pixels
- Normalization to [0, 1] range
- Data augmentation (training only):
  - Random rotation (±20°)
  - Random zoom (±10%)
  - Horizontal flip (50% probability)

## 🧠 Model Architecture

### Base Model: MobileNetV2

Our system uses **MobileNetV2** as the pre-trained base model:

```
Parameters: ~2.26M (frozen during initial training)
Architecture: Depthwise separable convolutions with inverted residuals
Input: 224×224×3 RGB images
Output: 1280-dimensional feature vector

Key Advantages:
- Efficient depthwise separable convolutions reduce computation
- Inverted residual blocks with linear bottlenecks
- Excellent feature extraction for medical images
- Fast inference suitable for deployment
- Robust performance on smaller datasets
- Pre-trained on ImageNet (1.4M images, 1000 classes)
```

**Why MobileNetV2?**
- Optimal balance between accuracy and efficiency
- Fewer parameters reduce overfitting on medical datasets
- Depthwise separable convolutions are highly effective
- Fast training and inference times
- Proven performance in medical imaging applications

### Custom Classification Head

```python
MobileNetV2 Base (Frozen initially, then fine-tuned)
    ↓
Global Average Pooling (1280 → 1280)
    ↓
Batch Normalization
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(256, ReLU)
    ↓
Batch Normalization
    ↓
Dropout(0.3)
    ↓
Dense(17, Softmax)  # Output probabilities for 17 classes
```

**Total Parameters**: ~3.05M (791K trainable, 2.26M in base)

### Key Design Decisions

**Global Average Pooling:**
- Reduces spatial dimensions while preserving features
- Much fewer parameters than Flatten
- More robust to spatial translations

**Batch Normalization:**
- Normalizes layer inputs
- Stabilizes training
- Allows higher learning rates
- Acts as regularization

**Dropout:**
- Prevents overfitting
- Forces network to learn robust features
- Different dropout rates (0.5, 0.3) for different layers

**ReLU Activation:**
- Non-linearity for learning complex patterns
- No vanishing gradient problem
- Computationally efficient

**Softmax Output:**
- Converts logits to probabilities
- Outputs sum to 1.0
- Ideal for multi-class classification

## 🎓 Training Pipeline

### Two-Stage Training Strategy

Our training employs a two-stage approach for optimal results:

#### Stage 1: Feature Extraction (20-30 epochs)

```python
# Freeze base model weights
base_model.trainable = False

# Train only classification head
# Learning Rate: 0.001
# Optimizer: Adam
```

**Purpose:**
- Let new classification layers learn appropriate features
- Prevent destroying pre-trained weights
- Faster convergence

**Expected Behavior:**
- Rapid initial improvement
- Training accuracy reaches ~85-90%
- Validation accuracy follows training

#### Stage 2: Fine-Tuning (10-20 epochs)

```python
# Unfreeze last N layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Fine-tune with lower learning rate
# Learning Rate: 0.0001
# Optimizer: Adam
```

**Purpose:**
- Adapt pre-trained features to medical images
- Learn domain-specific patterns
- Squeeze out final accuracy points

**Expected Behavior:**
- Slower but steady improvement
- Modest gains from fine-tuning (watch validation metrics)
- Slight overfitting is normal; use early stopping and LR schedule

### Loss Function

**Categorical Cross-Entropy:**

```
L = -Σᵢ yᵢ log(ŷᵢ)
```

Where:
- yᵢ: True label (one-hot encoded)
- ŷᵢ: Predicted probability

**Why This Loss?**
- Standard for multi-class classification
- Penalizes wrong predictions heavily
- Differentiable (enables gradient descent)
- Probabilistic interpretation

### Optimization

**Adam Optimizer:**
- Adaptive learning rates per parameter
- Combines momentum and RMSprop
- Beta1 = 0.9 (momentum)
- Beta2 = 0.999 (adaptive LR)
- Epsilon = 1e-8 (numerical stability)

**Learning Rate Schedule:**
- ReduceLROnPlateau callback
- Reduces LR by 0.5 when validation loss plateaus
- Patience: 5 epochs
- Minimum LR: 1e-7

### Regularization Techniques

1. **Dropout (0.5, 0.3)**
   - Randomly deactivates neurons during training
   - Prevents co-adaptation of features

2. **Batch Normalization**
   - Normalizes layer inputs
   - Reduces internal covariate shift

3. **Data Augmentation**
   - Rotation, zoom, flip
   - Artificially increases dataset size

4. **Early Stopping**
   - Monitors validation loss
   - Stops if no improvement for 10 epochs
   - Restores best weights

5. **L2 Regularization** (implicit in Adam)
   - Penalizes large weights
   - Encourages simpler models

### Training Callbacks

```python
ModelCheckpoint: Save best model
EarlyStopping: Prevent overfitting
ReduceLROnPlateau: Adaptive learning rate
TensorBoard: Visualization and logging
```

## 💻 Web Application

### Features

**1. Home Page**
- Project overview
- Key features and capabilities
- Quick statistics

**2. Detection Page**
- Image upload interface
- Real-time prediction
- Confidence scores for all classes
- Detailed pathology descriptions
- Interactive visualizations

**3. About Model Page**
- Architecture details
- Training strategy explanation
- Performance metrics
- Dataset information

### User Interface Design

**Color Scheme:**
- Primary: Blue (#1f77b4) - Professional, trustworthy
- Success: Green (#28a745) - Positive results
- Warning: Yellow (#ffc107) - Low confidence alerts
- Gradient backgrounds for emphasis

**Layout:**
- Responsive design (works on mobile)
- Two-column layouts for efficiency
- Card-based information presentation
- Interactive Plotly charts

**User Experience:**
- Clear navigation sidebar
- Progress indicators during processing
- Informative error messages
- Tooltips and help text
- Visual feedback for all actions

## 📁 Project Structure

```
neuropathology-dl/
├── app.py                          # Streamlit web application
├── train.py                        # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                     # Git ignore rules
│
├── src/                           # Source code
│   ├── models/
│   │   └── neuropathology_model.py # Model architecture
│   ├── data/
│   │   └── data_loader.py         # Data loading and preprocessing
│   └── utils/
│       └── helpers.py             # Utility functions
│
├── models/                        # Saved models (gitignored)
│   ├── .gitkeep
│   └── final_model.keras
│
├── assets/                        # Images and resources
│
└── logs/                          # Training logs (gitignored)
```

## 📈 Results

### Achieved Performance (On Test Set)

Trained on Kaggle Brain MRI 17-Class Dataset:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **78.13%** |
| Precision (weighted avg) | 82.33% |
| Recall (weighted avg) | 78.13% |
| F1-Score (weighted avg) | 77.44% |

### Top Performing Classes

| Class | F1-Score | Support |
|-------|----------|---------|
| NORMAL T1 | 97.56% | 41 |
| Neurocitoma T1C+ | 96.20% | 39 |
| Schwannoma T1C+ | 90.57% | 29 |
| Outros Tipos de Lesões T1 | 90.48% | 23 |
| Meningioma T1C+ | 89.90% | 94 |
| Glioma T1C+ | 89.44% | 76 |

**Key Observations:**
- Excellent detection of normal brain scans (97.56% F1-score)
- Strong performance on contrast-enhanced (T1C+) sequences
- Post-contrast imaging provides superior diagnostic information
- Balanced performance across 663 test samples

### Training Time

On GitHub Codespaces (4-core CPU):
- **Stage 1 (Feature Extraction)**: ~1.5 hours (10 epochs)
- **Stage 2 (Fine-Tuning)**: ~0.8 hours (5 epochs)
- **Total Training Time**: ~2.3 hours

On GPU (typical setup):
- **Total Training Time**: 20-40 minutes

### Model Size

- **Final Model**: ~28 MB (models/final_model.keras)
- **MobileNetV2 Base**: ~14 MB
- **Custom Head**: ~14 MB

## 🔮 Future Work

### Planned Enhancements

1. **Model Improvements**
   - Ensemble methods (combining multiple models)
   - Attention mechanisms for interpretability
   - Self-supervised pre-training on medical images
   - 3D CNN for volumetric MRI analysis

2. **Feature Additions**
   - Grad-CAM visualizations for explainability
   - Uncertainty quantification
   - Multi-modal fusion (T1, T2, FLAIR sequences)
   - Tumor segmentation capabilities
   - Longitudinal analysis (tracking over time)

3. **Deployment**
   - Docker containerization
   - REST API for integration
   - Mobile application
   - Cloud deployment (AWS/GCP/Azure)
   - Edge deployment for offline use

4. **User Experience**
   - DICOM file support
   - Batch processing
   - Report generation (PDF)
   - Integration with PACS systems
   - Multi-language support

5. **Research Extensions**
   - Federated learning for privacy
   - Active learning for efficient labeling
   - Few-shot learning for rare conditions
   - Adversarial robustness testing

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Code Improvements**
   - Bug fixes
   - Performance optimizations
   - Code refactoring
   - Test coverage

2. **Documentation**
   - Documentation improvements
   - Code comments
   - Usage examples

3. **Features**
   - New model architectures
   - Additional visualizations
   - UI enhancements
   - Data preprocessing techniques

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for **research and demonstration purposes only**. It should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

Key Points:
- Not FDA approved or clinically validated
- No warranty or guarantee of accuracy
- Not a substitute for professional medical diagnosis
- Use at your own risk
- Intended for learning and research

## 🙏 Acknowledgments

- **TensorFlow/Keras Team**: For the excellent deep learning framework
- **Streamlit Team**: For the intuitive web app framework
- **ImageNet**: For pre-trained model weights
- **Kaggle**: For hosting medical imaging datasets
- **Open Source Community**: For inspiration and resources

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- GitHub: [@amanraj41](https://github.com/amanraj41)
- Project Link: [https://github.com/amanraj41/neuropathology-dl](https://github.com/amanraj41/neuropathology-dl)

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐!

---

**Built with ❤️ as a practical reference implementation**