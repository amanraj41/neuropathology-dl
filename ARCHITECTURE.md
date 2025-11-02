# 🏗️ System Architecture Documentation

Technical overview of the Neuropathology Detection System architecture.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Model Architecture](#model-architecture)
4. [Grad-CAM Implementation](#grad-cam-implementation)
5. [Data Flow](#data-flow)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)

## System Overview

Modular, layered architecture following software engineering best practices:

- **Separation of Concerns**: Each module has a single responsibility
- **Modularity**: Components can be replaced or upgraded independently
- **Extensibility**: Easy to add new models, features, or visualizations
- **Maintainability**: Clear code structure with comprehensive documentation

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Web UI and user interaction |
| Backend | Python | Application logic |
| ML Framework | TensorFlow/Keras | Deep learning model |
| Data Processing | NumPy, Pillow, OpenCV | Image preprocessing |
| Visualization | Matplotlib, Plotly, Seaborn | Charts and Grad-CAM overlays |

## Architecture Layers

### 1. Presentation Layer (UI)

**File**: `app.py`

**Responsibilities**:
- User interface rendering
- Model selection and switching
- Input handling (image uploads, URL fetch)
- Grad-CAM controls (sensitivity, intensity sliders)
- Result visualization with color-coded classes
- Clinical information presentation

**Components**:
- Home page with project overview
- Diagnostic Classes page with 17 neuropathology categories
- Detection page with Grad-CAM integration
- About Model page with performance metrics

**Technologies**: Streamlit, Plotly, HTML/CSS, OpenCV (for Grad-CAM overlays)

### 2. Application Layer

**Files**: `train.py`, `validate.py`

**Responsibilities**:
- Training orchestration (two-stage transfer learning)
- Model evaluation and metrics computation
- Validation utilities
- CLI interfaces

### 3. Model Layer

**Directory**: `src/models/`

**Responsibilities**:
- Neural network architecture definition
- Model compilation and configuration
- Training callbacks
- Prediction interface
- Grad-CAM computation

**Key Classes**:
- `NeuropathologyModel`: Main model wrapper
  - `build_model()`: Constructs MobileNetV2 + custom head
  - `compile_model()`: Configures optimizer and loss
  - `fine_tune_model()`: Enables fine-tuning of last 20 layers
  - `predict()`: Makes predictions

### 4. Data Layer

**Directory**: `src/data/`

**Responsibilities**:
- Image loading and preprocessing
- Data augmentation
- Batch generation
- Dataset management

**Key Classes**:
- `MRIDataLoader`: Image preprocessing
  - `load_and_preprocess_image()`: Single image processing
  - Resizing to 224×224
  - Normalization to [0, 1]

### 5. Utility Layer

**Directory**: `src/utils/`

**Responsibilities**:
- Class name management
- Clinical descriptions
- MRI findings
- Helper functions

## Model Architecture

### MobileNetV2 Base

```
Input (224, 224, 3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
  - 154 layers
  - Depthwise separable convolutions
  - Inverted residual blocks
  - Output: 7×7×1280 feature maps
    ↓
Last Conv Layer: Conv_1 (7×7×1280) ← Used for Grad-CAM
    ↓
Global Average Pooling (1280)
```

### Custom Classification Head

```python
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

**Total Parameters**: ~3.05M
- Base model (frozen in stage 1): ~2.26M
- Classification head: ~0.79M
- Fine-tuning (stage 2): Last 20 layers unfrozen

## Grad-CAM Implementation

### Algorithm Overview

**Gradient-weighted Class Activation Mapping (Grad-CAM)**:
1. Forward pass through model
2. Compute gradients of predicted class w.r.t. last conv layer (Conv_1)
3. Global average pool gradients to get importance weights
4. Weight conv layer activations by importance
5. Apply ReLU and normalize to [0, 1]
6. Resize heatmap to input size (224×224)
7. Apply JET colormap and alpha blend with original image
8. Detect regions above threshold and draw contours

### Implementation in app.py

```python
def _compute_gradcam_overlay(self, img_array, predicted_class):
    # Get settings from sidebar
    threshold = st.session_state.get('gradcam_threshold', 0.5)
    alpha = st.session_state.get('gradcam_alpha', 0.35)
    
    # Build Grad-CAM model (base_model.input → [conv_output, final_output])
    gradcam_model = tf.keras.Model(
        inputs=base_model_input,
        outputs=[conv_layer_output, final_output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_tensor, training=False)
        class_channel = predictions[0, predicted_class]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Generate heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)
    
    # Resize and colorize
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype('uint8'), cv2.COLORMAP_JET)
    
    # Blend and draw contours
    overlay = cv2.addWeighted(base_img, 1.0 - alpha, heatmap_colored, alpha, 0)
    cv2.drawContours(overlay, contours_filtered, -1, (0, 255, 0), 2)
    
    return {'overlay_rgb': overlay_rgb, 'mask_any': regions_detected, 'debug_info': stats}
```

### Real-time Updates

**Key Innovation**: Cached `img_array` in `analysis_results`
- Store preprocessed image during initial analysis
- Regenerate Grad-CAM overlay when sliders change
- No model prediction needed (fast ~100ms vs ~1-2s)
- Seamless user experience

### Visual Output

- **Heatmap**: JET colormap (red=high activation, yellow=medium, blue=low)
- **Contours**: Green borders around regions above threshold
- **Alpha Blending**: Configurable transparency (0.1-0.8)
- **Sensitivity**: Threshold controls which regions are highlighted (0.1-0.9)

## Data Flow

### Training Flow

```
Dataset Directory
    ↓
[Load image paths and labels]
    ↓
[Train/Val/Test split (70/15/15)]
    ↓
Data Augmentation (training only)
  - Rotation ±20°
  - Zoom ±10%
  - Horizontal flip 50%
    ↓
Model.fit()
  ├─→ Stage 1: Feature Extraction (50 epochs, LR=0.001)
  │     - Freeze base model
  │     - Train classification head only
  └─→ Stage 2: Fine-Tuning (70 epochs, LR=0.0005)
        - Unfreeze last 20 layers
        - Fine-tune entire model
    ↓
[Callbacks]
  ├─→ ModelCheckpoint (save best based on val_accuracy)
  ├─→ EarlyStopping (patience=10, restore best weights)
  ├─→ ReduceLROnPlateau (factor=0.5, patience=5)
  └─→ TensorBoard (logging)
    ↓
Saved Models:
  - best_model.keras (Stage 1)
  - best_model_finetuned.keras (Stage 2)
  - final_model.keras (final snapshot)
  - class_names.json (labels)
  - metrics.json (test accuracy)
```

### Inference Flow

```
User Upload/URL (Web UI)
    ↓
Temporary Save
    ↓
MRIDataLoader.load_and_preprocess_image()
  ├─→ Resize to 224×224
  ├─→ Normalize to [0, 1]
  └─→ Convert to NumPy array
    ↓
np.expand_dims() [Add batch dimension]
    ↓
Model.predict()
  ├─→ Forward pass (no dropout)
  └─→ Softmax probabilities
    ↓
Store in session_state:
  - predictions (all 17 class probabilities)
  - predicted_class (argmax)
  - confidence (max probability)
  - class_name (from class_names.json)
  - img_array (for Grad-CAM caching)
    ↓
[If Grad-CAM enabled]
_compute_gradcam_overlay(img_array, predicted_class)
  ├─→ Build Grad-CAM model
  ├─→ Compute gradients
  ├─→ Generate heatmap
  ├─→ Detect regions above threshold
  └─→ Draw overlay with contours
    ↓
Display:
  - Predicted class with color coding
  - Confidence score with status (High/Moderate/Low)
  - Grad-CAM overlay (if regions detected)
  - Probability distribution chart (Plotly)
  - Clinical description
  - MRI findings
```

## Training Pipeline

### Two-Stage Strategy

#### Stage 1: Feature Extraction (50 epochs)

**Configuration**:
```python
base_model.trainable = False  # Freeze all MobileNetV2 layers
learning_rate = 0.001
optimizer = Adam
batch_size = 16
```

**Purpose**: Train classification head without destroying pre-trained weights

#### Stage 2: Fine-Tuning (70 epochs)

**Configuration**:
```python
# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

learning_rate = 0.0005  # Lower LR crucial!
optimizer = Adam
batch_size = 16
```

**Purpose**: Adapt pre-trained features to medical imaging domain

### Loss Function

**Categorical Cross-Entropy**:
```
L = -Σᵢ yᵢ log(ŷᵢ)

where:
- yᵢ: true label (one-hot encoded)
- ŷᵢ: predicted probability
```

### Regularization

1. **Dropout (0.5, 0.3)**: Prevents overfitting
2. **Batch Normalization**: Stabilizes training, acts as regularization
3. **Data Augmentation**: Increases dataset diversity
4. **Early Stopping**: Monitors val_loss, patience=10
5. **L2 Regularization**: Implicit in Adam optimizer

## Inference Pipeline

### Prediction Steps

1. **Preprocessing**: Resize (224×224), normalize [0, 1], add batch dimension
2. **Forward Pass**: model.predict() with dropout disabled
3. **Post-processing**: argmax for class, max for confidence
4. **Grad-CAM** (if enabled): Compute heatmap, detect regions, draw overlay
5. **Visualization**: Plotly charts, color-coded classes, clinical info

### Performance

**Inference Time** (CPU):
- Model prediction: ~1-2 seconds
- Grad-CAM computation: ~100-200ms
- Total: ~1.2-2.2 seconds

**Memory Usage**:
- Model: ~500 MB
- Single image: ~6 MB
- Grad-CAM temp tensors: ~50 MB
- **Total**: ~556 MB

## Key Design Decisions

### 1. Why MobileNetV2?

- Efficient depthwise separable convolutions
- Small parameter count (~2.26M base)
- Fast inference suitable for web deployment
- Proven performance on medical images
- Pre-trained on ImageNet (strong feature extractor)

### 2. Why Two-Stage Training?

- Prevents destroying pre-trained weights
- New layers stabilize first before fine-tuning
- Better final accuracy (8-10% improvement typical)
- More stable training process

### 3. Why Grad-CAM?

- Provides visual explanations for model decisions
- Builds trust through transparency
- Helps validate predictions
- Useful for identifying potential tumor locations
- Real-time updates enhance user experience

### 4. Why Global Average Pooling?

- Much fewer parameters than Flatten (prevents overfitting)
- Spatial invariance (position doesn't matter)
- Better generalization
- Standard in modern architectures

### 5. Why Batch Normalization?

- Normalizes activations between layers
- Allows higher learning rates
- Faster convergence
- Acts as regularization

### 6. Why Dropout?

- Simple and effective regularization
- Forces redundant representations
- Prevents co-adaptation of features
- Critical for small medical datasets

### 7. Why Real-time Grad-CAM Updates?

- Caching preprocessed img_array avoids re-prediction
- Slider changes update overlay instantly (~100ms)
- Better user experience
- Allows experimentation with different thresholds

## Extensibility

### Adding New Features

1. **Tumor Segmentation**: Extend Grad-CAM to produce binary masks
2. **Multi-modal Fusion**: Combine T1, T1C+, T2 sequences
3. **3D CNN**: Process volumetric MRI data
4. **Uncertainty Quantification**: Add dropout at inference for confidence intervals
5. **Ensemble Models**: Combine multiple architectures for better accuracy

### Deployment Options

1. **Docker Container**: Package entire app with dependencies
2. **Cloud Platforms**: Deploy on AWS SageMaker, Google AI Platform, Azure ML
3. **Mobile/Edge**: Convert to TensorFlow Lite, optimize with quantization
4. **REST API**: Create Flask/FastAPI backend for integration

## Conclusion

This architecture provides:
- **Modularity**: Easy to modify and extend
- **Performance**: Fast inference (~1-2s on CPU)
- **Explainability**: Grad-CAM visual explanations
- **Usability**: Interactive web interface
- **Maintainability**: Clear code structure

The design balances simplicity with power, making it suitable for production deployment and further research.

---

**Version 1.1 | © 2025**
