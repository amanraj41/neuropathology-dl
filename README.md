# 🧠 Neuropathology Detection System

A production-ready deep learning system for detecting neuropathological conditions from brain MRI scans with Grad-CAM explainability for anomaly localization.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)

**Created by**: Aman Raj | **Version**: 1.1 | **Year**: 2025

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

A comprehensive neuropathology detection system powered by deep learning for medical MRI analysis. Features:

- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet, fine-tuned for medical imaging
- **17-Class Classification**: Gliomas, meningiomas, schwannomas, neurocytomas, lesions & normal scans
- **Grad-CAM Explainability**: Visual explanations with real-time anomaly localization overlays
- **Interactive Web Interface**: Streamlit application with model selection and real-time predictions
- **Production-Ready**: Pre-trained models with 78.13% test accuracy included

### Detected Conditions

The system classifies brain MRI scans into **17 distinct categories** across multiple imaging modalities (T1, T1C+, T2):

**Primary Tumor Categories:**
- **Glioma** (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) - T1, T1C+, T2
- **Meningioma** (de Baixo Grau, Atípico, Anaplásico, Transicional) - T1, T1C+, T2
- **NORMAL** - T1, T2
- **Neurocytoma** (Central - Intraventricular, Extraventricular) - T1, T1C+, T2
- **Other Lesions** (Abscesses, Cysts, Encephalopathies) - T1, T1C+, T2
- **Schwannoma** (Acustico, Vestibular - Trigeminal) - T1, T1C+, T2

Each class includes detailed clinical information and characteristic MRI findings.

## ✨ Features

### Core Features

- **MobileNetV2 Architecture**: Efficient depthwise separable convolutions optimized for medical imaging
- **Two-Stage Training**: Feature extraction (50 epochs) + Fine-tuning (70 epochs)
- **Data Augmentation**: Rotation, zoom, flip transformations for robust training
- **Grad-CAM Visualization**: Real-time anomaly localization with adjustable sensitivity and intensity
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score with per-class breakdowns
- **Pre-trained Models**: Best finetuned model achieves 78.13% test accuracy (ready to deploy)

### User Interface Features

- **Model Selection & Switching**: Load and compare different trained models dynamically
- **Real-time Predictions**: Upload MRI images or fetch from URL for instant diagnosis
- **Grad-CAM Overlays**: Interactive heatmaps highlighting suspicious regions with green contours
- **Adjustable Detection**: Sensitivity and intensity sliders for fine-tuning anomaly localization
- **Test Accuracy Display**: View model performance metrics directly in the UI
- **Confidence Scores**: Detailed probability distributions for all 17 classes with color-coded visualization
- **Clinical Information**: Comprehensive medical descriptions and MRI findings for each pathology
- **Interactive Visualizations**: Plotly charts for prediction analysis
- **Responsive Layout**: Works on desktop and mobile devices

## 🚀 Quick Start

### Option 1: Use Pre-trained Models (Recommended)

1. **Clone the repository:**
\`\`\`bash
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
\`\`\`

2. **Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. **Run the app:**
\`\`\`bash
streamlit run app.py
\`\`\`

The app includes pre-trained models in the \`models/\` directory ready for immediate use!

### Option 2: Train Your Own Model

If you want to experiment with different configurations:

1. **Clone and install** (as above)

2. **Prepare your dataset** (see [Dataset](#dataset) section)

3. **Train the model:**
\`\`\`bash
python train.py --data_dir ./data/brain_mri_17 --base_model mobilenet --epochs_stage1 50 --epochs_stage2 70
\`\`\`

4. **Run the app:**
\`\`\`bash
streamlit run app.py
\`\`\`

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the Repository

\`\`\`bash
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl
\`\`\`

### Step 2: Create Virtual Environment (Recommended)

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On Linux/Mac:
source venv/bin/activate
\`\`\`

### Step 3: Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 4: Verify Installation

\`\`\`bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit installed successfully')"
\`\`\`

## 📖 Usage

### Running the Web Application

Start the Streamlit web interface:

\`\`\`bash
streamlit run app.py
\`\`\`

The application will open in your default browser at \`http://localhost:8501\`.

**Features Available:**
- Load pre-trained models from dropdown
- Upload MRI images or provide URLs
- Enable/disable Grad-CAM anomaly localization
- Adjust detection sensitivity (threshold)
- Control overlay intensity (alpha)
- View color-coded predictions with confidence scores
- Access clinical descriptions and MRI findings

### Training a Model

To train a model on your own dataset:

\`\`\`bash
python train.py --data_dir /path/to/your/dataset --epochs_stage1 50 --epochs_stage2 70
\`\`\`

**Training Arguments:**

\`\`\`bash
python train.py \\
    --data_dir ./data/brain_mri_17 \\    # Path to dataset
    --base_model mobilenet \\            # Base architecture
    --batch_size 16 \\                   # Batch size (16 for CPU)
    --epochs_stage1 50 \\                # Feature extraction epochs
    --epochs_stage2 70 \\                # Fine-tuning epochs
    --learning_rate 0.001 \\             # Initial learning rate
    --learning_rate_finetune 0.0005 \\   # Fine-tuning learning rate
    --trainable_layers 20               # Layers to fine-tune
\`\`\`

The training script saves:
- \`models/best_model.keras\` - Best Stage 1 model
- \`models/best_model_finetuned.keras\` - Best Stage 2 model
- \`models/final_model.keras\` - Final snapshot
- \`models/class_names.json\` - Class labels for dynamic UI
- \`models/metrics.json\` - Test accuracy and performance metrics

### Making Predictions

#### Using Python Script

\`\`\`python
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
\`\`\`

## 📊 Dataset

### Expected Dataset Structure

\`\`\`
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
\`\`\`

### Recommended Dataset

**Kaggle Brain MRI 17-Class Dataset**
- **17 neuropathology classes** across multiple MRI modalities
- **Comprehensive clinical spectrum**: Primary tumors, benign lesions, normal scans
- **Multiple imaging modalities**: T1, T1 post-contrast (T1C+), T2-weighted sequences
- **Pre-processed and ready to use**

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

\`\`\`
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
\`\`\`

### Custom Classification Head

\`\`\`python
MobileNetV2 Base (Pre-trained, fine-tuned in stage 2)
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
\`\`\`

**Total Parameters**: ~3.05M (791K trainable in stage 1, more in stage 2 after unfreezing)

### Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping (Grad-CAM):**
- Generates visual explanations for CNN decisions
- Highlights regions that contribute most to predictions
- Uses gradients flowing into final convolutional layer (Conv_1)
- Produces class-discriminative localization maps
- Real-time overlay with adjustable sensitivity and intensity

**Implementation:**
- Target layer: Conv_1 (last convolutional layer, 7×7×1280)
- Heatmap colorization: JET colormap (red=high, blue=low)
- Region detection: Binary thresholding + contour detection
- Visual output: Alpha-blended overlay with green contour borders

## �� Training Pipeline

### Two-Stage Training Strategy

#### Stage 1: Feature Extraction (50 epochs)

\`\`\`python
# Freeze base model weights
base_model.trainable = False

# Train only classification head
# Learning Rate: 0.001
# Optimizer: Adam
\`\`\`

**Purpose:**
- Let new classification layers learn appropriate features
- Prevent destroying pre-trained weights
- Faster convergence

#### Stage 2: Fine-Tuning (70 epochs)

\`\`\`python
# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Fine-tune with lower learning rate
# Learning Rate: 0.0005
# Optimizer: Adam
\`\`\`

**Purpose:**
- Adapt pre-trained features to medical images
- Learn domain-specific patterns
- Achieve optimal accuracy

### Regularization Techniques

1. **Dropout (0.5, 0.3)** - Prevents overfitting
2. **Batch Normalization** - Stabilizes training
3. **Data Augmentation** - Increases dataset diversity
4. **Early Stopping** - Monitors validation loss (patience=10)
5. **L2 Regularization** - Implicit in Adam optimizer

### Training Callbacks

\`\`\`python
ModelCheckpoint: Save best model based on validation accuracy
EarlyStopping: Prevent overfitting, restore best weights
ReduceLROnPlateau: Adaptive learning rate (factor=0.5, patience=5)
TensorBoard: Visualization and logging
\`\`\`

## 💻 Web Application

### Features

**1. Home Page**
- Project overview with Grad-CAM mention
- Key features and capabilities
- Quick statistics and model information

**2. Diagnostic Classes Page**
- Complete list of 17 neuropathology classes
- Color-coded class visualization
- MRI modality explanations (T1, T1C+, T2)
- Clinical resources and references
- Grad-CAM explainability mention

**3. Detection Page**
- Model selection dropdown with test accuracy display
- Image upload interface or URL fetch
- **Grad-CAM Controls:**
  - Enable/disable toggle
  - Detection sensitivity slider (0.1-0.9)
  - Overlay intensity slider (0.1-0.8)
- Real-time prediction with confidence scores
- Anomaly localization overlay with colored heatmap and green contours
- Detailed probability distribution for all classes
- Clinical descriptions and MRI findings

**4. About Model Page**
- MobileNetV2 architecture details
- Two-stage training strategy explanation
- Performance metrics (accuracy, precision, recall, F1-score)
- Grad-CAM explainability section
- Training hardware specifications

### Grad-CAM Visualization

**What You See:**
- **Heatmap Overlay**: Red/yellow areas indicate high model activation (suspicious regions), blue areas indicate low activation
- **Green Contours**: Boundaries of detected anomalous regions above the sensitivity threshold
- **Normal Scans**: No overlay displayed for NORMAL class predictions
- **Real-time Updates**: Adjust sensitivity and intensity sliders without re-analyzing

**Use Cases:**
- Validate model decisions by seeing what regions influenced prediction
- Identify potential tumor locations
- Compare suspicious regions across different MRI modalities
- Build trust through model transparency

## 📁 Project Structure

\`\`\`
neuropathology-dl/
├── app.py                          # Streamlit web application (Grad-CAM integrated)
├── train.py                        # Model training script
├── validate.py                     # Project validation script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ARCHITECTURE.md                 # System architecture documentation
├── QUICKSTART.md                   # Quick start guide
├── LICENSE                         # MIT License
├── .gitignore                     # Git ignore rules
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── neuropathology_model.py # Model architecture
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py         # Data loading and preprocessing
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
│
├── models/                        # Saved models (included in repo)
│   ├── .gitkeep
│   ├── best_model_finetuned.keras # Pre-trained finetuned model
│   ├── class_names.json           # Class label mappings
│   └── metrics.json               # Model performance metrics
│
├── scripts/                       # Utility scripts
│   ├── check_training.sh
│   ├── evaluate_models.py
│   └── post_training_evaluation.sh
│
└── data/                          # Dataset directory (excluded from repo)
    └── brain_mri_17/              # 17-class MRI dataset
\`\`\`

## 📈 Results

### Achieved Performance (Test Set)

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
| Neurocytoma T1C+ | 96.20% | 39 |
| Schwannoma T1C+ | 90.57% | 29 |
| Other Lesions T1 | 90.48% | 23 |
| Meningioma T1C+ | 89.90% | 94 |
| Glioma T1C+ | 89.44% | 76 |

**Key Observations:**
- Excellent detection of normal brain scans (97.56% F1-score)
- Strong performance on contrast-enhanced (T1C+) sequences
- Post-contrast imaging provides superior diagnostic information
- Balanced performance across 663 test samples

### Grad-CAM Validation

With real Glioma MRI test image:
- **Threshold 0.3**: 28,713 pixels highlighted, 2 regions detected
- **Threshold 0.5**: 10,709 pixels highlighted, 3 regions detected  
- **Threshold 0.7**: 3,176 pixels highlighted, 1 region detected
- **Confidence**: 96.94% for correct class prediction

### Training Time

On GitHub Codespaces (4-core CPU):
- **Total Training Time**: ~2-3 hours for 120 epochs (both stages)

On GPU (typical setup):
- **Total Training Time**: 20-40 minutes

### Model Size

- **Final Model**: ~28 MB (models/best_model_finetuned.keras)
- **MobileNetV2 Base**: ~14 MB
- **Custom Head**: ~14 MB

## �� Contributing

Contributions are welcome! Areas for contribution:

1. **Model Improvements**
   - Additional architectures
   - Ensemble methods
   - Improved Grad-CAM variants (Grad-CAM++, Score-CAM)

2. **Features**
   - Tumor segmentation
   - Multi-modal fusion
   - DICOM support
   - Batch processing

3. **Deployment**
   - Docker containerization
   - REST API
   - Mobile optimization

4. **Documentation**
   - Usage examples
   - Video tutorials
   - API documentation

### Contribution Process

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for **research and demonstration purposes only**. It should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

Key Points:
- Not FDA approved or clinically validated
- No warranty or guarantee of accuracy
- Not a substitute for professional medical diagnosis
- Use at your own risk

## 🙏 Acknowledgments

- **TensorFlow/Keras Team**: Deep learning framework
- **Streamlit Team**: Web app framework
- **ImageNet**: Pre-trained model weights
- **Kaggle**: Medical imaging datasets
- **Grad-CAM Authors**: Selvaraju et al. for explainability technique

## �� Contact

- GitHub: [@amanraj41](https://github.com/amanraj41)
- Project Link: [https://github.com/amanraj41/neuropathology-dl](https://github.com/amanraj41/neuropathology-dl)

---

**Built with ❤️ | Version 1.1 | © 2025**
