# 🎓 Comprehensive Training Instructions for Neuropathology Detection Model

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Parameters](#training-parameters)
- [Recommended Training Configurations](#recommended-training-configurations)
- [Step-by-Step Training Guide](#step-by-step-training-guide)
- [Model Evaluation](#model-evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

This guide provides comprehensive instructions for training the neuropathology detection model to achieve the best possible accuracy. The training process uses a two-stage approach:
1. **Stage 1**: Feature extraction with frozen base model
2. **Stage 2**: Fine-tuning with unfrozen layers

## Prerequisites

### Hardware Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, V100, A100)
- **Minimum**: 4-core CPU with 16GB RAM (training will be significantly slower)

### Software Requirements
- Python 3.8 or higher
- TensorFlow 2.16+ with Keras 3.0+
- CUDA 11.8+ and cuDNN 8.6+ (for GPU training)
- All dependencies from `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/amanraj41/neuropathology-dl.git
cd neuropathology-dl

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

## Dataset Preparation

### Download the Kaggle Dataset

The model is designed for the **Brain MRI 17-Class Dataset** available on Kaggle:
- **Dataset URL**: https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes
- **Size**: ~2.5GB
- **Classes**: 17 neuropathology classes across T1, T1C+, and T2 modalities

### Download Steps

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Configure Kaggle API Credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download Dataset**:
```bash
# Download and extract
kaggle datasets download -d fernando2rad/brain-tumor-mri-images-17-classes
unzip brain-tumor-mri-images-17-classes.zip -d data/brain_mri_17

# Verify structure
ls data/brain_mri_17/
```

### Expected Directory Structure

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

## Training Parameters

### Key Parameters Explained

#### Stage 1: Feature Extraction
- `--epochs_stage1`: Number of epochs for initial training (recommended: 30-50)
- `--learning_rate`: Initial learning rate (recommended: 0.001)
- Base model layers are **frozen** during this stage

#### Stage 2: Fine-Tuning
- `--epochs_stage2`: Number of epochs for fine-tuning (recommended: 20-30)
- `--learning_rate_finetune`: Lower learning rate for fine-tuning (recommended: 0.0001 - 0.00001)
- `--trainable_layers`: Number of layers to unfreeze (recommended: 20-30)

#### Other Important Parameters
- `--batch_size`: Number of samples per batch (GPU: 32-64, CPU: 8-16)
- `--base_model`: Pre-trained architecture (options: mobilenet, efficientnet, resnet, vgg)
- `--data_dir`: Path to dataset directory

### Why These Parameters Matter

**Higher Epochs**:
- More epochs allow the model to learn better representations
- Risk of overfitting increases with too many epochs
- Use early stopping to prevent overfitting

**Lower Learning Rate for Fine-Tuning**:
- Prevents destroying pre-trained weights
- Allows subtle adjustments to features
- Improves final accuracy without overfitting

**Batch Size**:
- Larger batch size: More stable gradients, faster training, requires more memory
- Smaller batch size: Noisier gradients, better generalization, less memory

## Recommended Training Configurations

### Configuration 1: Maximum Accuracy (GPU Required)

**Best for**: Achieving highest possible accuracy, research purposes

```bash
python train.py \
    --data_dir data/brain_mri_17 \
    --base_model efficientnet \
    --batch_size 32 \
    --epochs_stage1 50 \
    --epochs_stage2 30 \
    --learning_rate 0.001 \
    --learning_rate_finetune 0.00005 \
    --trainable_layers 30
```

**Expected Results**:
- Test Accuracy: 82-85%
- Training Time: 3-5 hours on V100 GPU
- Model Size: ~30 MB

### Configuration 2: Balanced Performance (GPU Recommended)

**Best for**: Good accuracy with reasonable training time

```bash
python train.py \
    --data_dir data/brain_mri_17 \
    --base_model mobilenet \
    --batch_size 32 \
    --epochs_stage1 30 \
    --epochs_stage2 20 \
    --learning_rate 0.001 \
    --learning_rate_finetune 0.0001 \
    --trainable_layers 20
```

**Expected Results**:
- Test Accuracy: 78-82%
- Training Time: 2-3 hours on V100 GPU
- Model Size: ~28 MB

### Configuration 3: Quick Training (CPU Compatible)

**Best for**: Testing, development, limited resources

```bash
python train.py \
    --data_dir data/brain_mri_17 \
    --base_model mobilenet \
    --batch_size 16 \
    --epochs_stage1 10 \
    --epochs_stage2 5 \
    --learning_rate 0.001 \
    --learning_rate_finetune 0.0001 \
    --trainable_layers 20
```

**Expected Results**:
- Test Accuracy: 70-75%
- Training Time: 2-3 hours on 4-core CPU
- Model Size: ~28 MB

### Configuration 4: Ultra High Accuracy (Requires Powerful GPU)

**Best for**: Research, competitions, production deployment

```bash
python train.py \
    --data_dir data/brain_mri_17 \
    --base_model efficientnet \
    --batch_size 64 \
    --epochs_stage1 60 \
    --epochs_stage2 40 \
    --learning_rate 0.001 \
    --learning_rate_finetune 0.00003 \
    --trainable_layers 40
```

**Expected Results**:
- Test Accuracy: 84-87%
- Training Time: 5-8 hours on A100 GPU
- Model Size: ~30 MB

## Step-by-Step Training Guide

### Step 1: Prepare Dataset

```bash
# Ensure dataset is downloaded and extracted
ls data/brain_mri_17/

# Check number of images per class
for dir in data/brain_mri_17/*/; do
    echo "$(basename "$dir"): $(find "$dir" -type f | wc -l) images"
done
```

### Step 2: Start Training

Choose a configuration from above and run:

```bash
# Example: Balanced performance configuration
python train.py \
    --data_dir data/brain_mri_17 \
    --base_model mobilenet \
    --batch_size 32 \
    --epochs_stage1 30 \
    --epochs_stage2 20 \
    --learning_rate 0.001 \
    --learning_rate_finetune 0.0001 \
    --trainable_layers 20
```

### Step 3: Monitor Training

**Watch Training Output**:
- Epoch progress
- Training/validation loss
- Training/validation accuracy
- Learning rate changes
- Early stopping triggers

**Check TensorBoard** (optional):
```bash
# In a separate terminal
tensorboard --logdir logs/
# Open http://localhost:6006 in browser
```

### Step 4: Training Completion

After training completes, you'll find:
- `models/best_model.keras` - Best model from Stage 1
- `models/best_model_finetuned.keras` - Best model from Stage 2 (if fine-tuning enabled)
- `models/final_model.keras` - Final trained model
- `models/training_history.png` - Training curves
- `models/confusion_matrix.png` - Confusion matrix
- `models/class_names.json` - Class names
- `models/metrics.json` - Model metrics

## Model Evaluation

### Evaluate Trained Models

After training, evaluate each model to generate comprehensive metrics:

```bash
# Evaluate best_model.keras
python evaluate_model.py \
    --model_path models/best_model.keras \
    --data_dir data/brain_mri_17

# Evaluate finetuned model
python evaluate_model.py \
    --model_path models/best_model_finetuned.keras \
    --data_dir data/brain_mri_17

# Evaluate final model
python evaluate_model.py \
    --model_path models/final_model.keras \
    --data_dir data/brain_mri_17
```

This generates:
- `models/best_model_metrics.json`
- `models/best_model_finetuned_metrics.json`
- `models/final_model_metrics.json`

### Using Metrics in Web App

The metrics JSON files are automatically loaded by `app.py` to display:
- Test accuracy
- Per-class precision, recall, F1-score
- Model performance statistics

## Troubleshooting

### Out of Memory (OOM) Errors

**GPU OOM**:
```bash
# Reduce batch size
python train.py --batch_size 16 ...  # or 8

# Use smaller base model
python train.py --base_model mobilenet ...
```

**CPU OOM**:
```bash
# Reduce batch size significantly
python train.py --batch_size 8 ...  # or 4
```

### Slow Training on CPU

```bash
# Use smaller batch size and fewer epochs
python train.py \
    --batch_size 8 \
    --epochs_stage1 10 \
    --epochs_stage2 5 \
    --base_model mobilenet \
    --data_dir data/brain_mri_17
```

### Overfitting (High Training Accuracy, Low Validation Accuracy)

**Solutions**:
1. Reduce number of epochs
2. Increase dropout rates (edit `src/models/neuropathology_model.py`)
3. Reduce number of trainable layers
4. Add more data augmentation

### Underfitting (Low Training and Validation Accuracy)

**Solutions**:
1. Increase number of epochs
2. Increase learning rate slightly
3. Reduce dropout rates
4. Use more powerful base model (efficientnet, resnet)
5. Increase number of trainable layers

### Model Not Improving

**Checklist**:
- [ ] Check if data is loaded correctly
- [ ] Verify class balance
- [ ] Try different learning rates
- [ ] Check for data preprocessing issues
- [ ] Monitor training curves in TensorBoard
- [ ] Ensure early stopping patience is appropriate

## Advanced Tips

### 1. Learning Rate Scheduling

For even better results, consider implementing cosine annealing or exponential decay.

### 2. Ensemble Methods

Train multiple models with different configurations and ensemble their predictions:
```python
# Average predictions from multiple models
predictions = (model1.predict(x) + model2.predict(x) + model3.predict(x)) / 3
```

### 3. Class Weights

If dataset is imbalanced, use class weights:
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# Pass to model.fit(class_weight=class_weights, ...)
```

### 4. Mixed Precision Training (GPU)

For faster training on modern GPUs:
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

## Best Practices Summary

1. **Always use GPU for training** if possible
2. **Start with recommended configurations** and adjust based on results
3. **Monitor validation metrics** - stop if overfitting occurs
4. **Use TensorBoard** for detailed training insights
5. **Evaluate multiple models** - keep the best performing one
6. **Save checkpoints frequently** - don't lose progress
7. **Document your experiments** - track parameters and results
8. **Test on held-out data** - evaluate real-world performance

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [Medical Image Analysis Papers](https://paperswithcode.com/task/medical-image-analysis)
- [Data Augmentation Techniques](https://www.tensorflow.org/tutorials/images/data_augmentation)

---

**Happy Training! 🚀**
