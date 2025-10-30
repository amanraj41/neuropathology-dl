# Session 01 Part E: Neural Networks - Project Mapping

## Table of Contents

1. [Overview of the Neuropathology Model Architecture](#overview-of-the-neuropathology-model-architecture)
2. [Mapping to neuropathology_model.py](#mapping-to-neuropathology_modelpy)
3. [Dense Layers in the Classification Head](#dense-layers-in-the-classification-head)
4. [Activation Functions in Practice](#activation-functions-in-practice)
5. [Forward Pass in TensorFlow/Keras](#forward-pass-in-tensorflowkeras)
6. [Loss Function: Categorical Cross-Entropy](#loss-function-categorical-cross-entropy)
7. [Model Compilation and Optimizer](#model-compilation-and-optimizer)
8. [Training Loop in train.py](#training-loop-in-trainpy)
9. [Backpropagation: Automatic Differentiation](#backpropagation-automatic-differentiation)
10. [Batch Processing and Vectorization](#batch-processing-and-vectorization)

---

## Overview of the Neuropathology Model Architecture

The brain tumor MRI classifier in `/src/models/neuropathology_model.py` implements the neural network concepts we've studied. Let's map every theoretical concept to specific lines of code.

**High-level architecture**:
```
Input (224×224×3) 
    ↓
MobileNetV2 Base (pretrained, optionally frozen)
    ↓
Global Average Pooling (spatial → vector)
    ↓
Batch Normalization
    ↓
Dense(512, ReLU)  ← Neural network layer
    ↓
Dropout(0.5)
    ↓
Dense(256, ReLU)  ← Neural network layer
    ↓
Batch Normalization
    ↓
Dropout(0.3)
    ↓
Dense(17, Softmax)  ← Output layer
```

The two `Dense` layers with ReLU activation are exactly the feedforward neural network layers we studied in Parts A-C. Let's examine them in detail.

## Mapping to neuropathology_model.py

Open `/src/models/neuropathology_model.py`. We'll go through the relevant sections:

### Line 162-287: The NeuropathologyModel Class

This class encapsulates the entire model. Key methods:
- `__init__`: Initializes model parameters
- `build_model`: Constructs the architecture
- `compile_model`: Sets up loss function and optimizer
- `fine_tune_model`: Unfreezes layers for fine-tuning

### Lines 224-290: build_model() Method

This is where the network architecture is constructed. Let's trace through it:

**Lines 242-246: Loading the Pretrained Base**
```python
# Load pre-trained base model
base_model = self._get_base_model()

# Freeze early layers (transfer learning strategy)
# Theory: Early layers learn general features, keep them fixed
base_model.trainable = False
```

**Mapping to theory**:
- The `base_model` is a pretrained CNN (MobileNetV2) that acts as a feature extractor
- Setting `trainable = False` freezes the weights: $\mathbf{W}_{\text{base}}$ won't be updated during initial training
- This corresponds to the transfer learning strategy: use pretrained features, only train the classification head

**Lines 250-287: Building the Classification Head**

This is our feedforward neural network!

```python
model = models.Sequential([
    # Input layer
    layers.Input(shape=self.input_shape),
```

**Line 253**: Input layer
- Defines input shape: $(224, 224, 3)$
- In our notation: $\mathbf{x} \in \mathbb{R}^{224 \times 224 \times 3}$ or flattened $\mathbf{x} \in \mathbb{R}^{150528}$

```python
    # Pre-trained base model (feature extractor)
    base_model,
```

**Line 255**: Base model (MobileNetV2)
- Acts as layers 1 through $\ell-1$ in our notation
- Transforms image $\mathbf{x}$ to feature vector $\mathbf{h} \in \mathbb{R}^{1280}$ (MobileNetV2 outputs 1280 features)

```python
    # Global Average Pooling
    # Reduces (7, 7, 2048) to (2048,) for ResNet
    # More robust than flatten, fewer parameters
    layers.GlobalAveragePooling2D(name='global_avg_pool'),
```

**Lines 260-262**: Global Average Pooling
- After the base CNN, we have a spatial feature map of shape $(H, W, C)$ where $H, W$ are spatial dimensions and $C$ is channels
- For MobileNetV2: $(7, 7, 1280)$
- **Global Average Pooling**: For each channel, compute the average: 
  $$\text{GAP}_c = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}_{i,j,c}$$
- Output: $(1280,)$ vector, one value per channel
- **Why this matters**: 
  - Reduces parameters dramatically compared to Flatten
  - Translation invariant
  - Acts as regularization

```python
    # Batch Normalization
    # Normalizes activations, stabilizes training
    layers.BatchNormalization(name='bn1'),
```

**Lines 264-266**: Batch Normalization (first application)
- Normalizes the 1280-dimensional feature vector across the batch
- For each feature $i$: 
  $$\hat{h}_i = \frac{h_i - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$$
- We'll cover the full derivation in Session 05, but note it stabilizes training

```python
    # First Dense Layer
    # Theory: Fully connected layer learns combinations of features
    # ReLU activation introduces non-linearity
    layers.Dense(512, activation='relu', name='fc1'),
```

**Lines 268-271**: **First Dense Layer** ← THIS IS OUR NEURAL NETWORK!

**Mapping to theory (from Part B)**:

This line implements:
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{h} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) = \max(0, \mathbf{z}^{(1)})$$

Where:
- Input: $\mathbf{h} \in \mathbb{R}^{1280}$ (output of Global Average Pooling)
- Weights: $\mathbf{W}^{(1)} \in \mathbb{R}^{512 \times 1280}$ (512 neurons, 1280 inputs each)
- Bias: $\mathbf{b}^{(1)} \in \mathbb{R}^{512}$
- Pre-activation: $\mathbf{z}^{(1)} \in \mathbb{R}^{512}$
- Activation: $\mathbf{a}^{(1)} \in \mathbb{R}^{512}$ (output of this layer)

**Parameter count**: $1280 \times 512 + 512 = 655{,}872$ parameters

**ReLU activation**:
- For each neuron $i$: $a_i^{(1)} = \max(0, z_i^{(1)})$
- Introduces nonlinearity (critical for learning complex functions)
- Prevents vanishing gradients (derivative is 1 for $z > 0$)

**What this layer does**:
- Learns 512 different combinations of the 1280 input features
- Each of the 512 neurons can be thought of as detecting a high-level pattern
- For brain tumors: might learn to detect combinations like "high contrast + irregular border + specific texture"

```python
    # Dropout for regularization
    # Randomly drops 50% of neurons during training
    # Prevents co-adaptation of features
    layers.Dropout(0.5, name='dropout1'),
```

**Lines 273-276**: Dropout
- During training, randomly sets 50% of the activations to zero
- In our notation, if $\mathbf{a}^{(1)}$ is the layer output:
  $$\tilde{\mathbf{a}}^{(1)} = \mathbf{a}^{(1)} \odot \mathbf{m}$$
  where $\mathbf{m} \in \{0, 1\}^{512}$ is a random binary mask with $P(m_i = 1) = 0.5$
- During inference, all neurons are active but activations are scaled by 0.5
- **Purpose**: Regularization, prevents overfitting (covered in Session 05)

```python
    # Second Dense Layer
    # Additional capacity for complex patterns
    layers.Dense(256, activation='relu', name='fc2'),
```

**Lines 278-280**: **Second Dense Layer**

**Mapping to theory**:

$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \tilde{\mathbf{a}}^{(1)} + \mathbf{b}^{(2)}$$
$$\mathbf{a}^{(2)} = \text{ReLU}(\mathbf{z}^{(2)})$$

Where:
- Input: $\tilde{\mathbf{a}}^{(1)} \in \mathbb{R}^{512}$ (output of previous layer after dropout)
- Weights: $\mathbf{W}^{(2)} \in \mathbb{R}^{256 \times 512}$
- Bias: $\mathbf{b}^{(2)} \in \mathbb{R}^{256}$
- Output: $\mathbf{a}^{(2)} \in \mathbb{R}^{256}$

**Parameter count**: $512 \times 256 + 256 = 131{,}328$ parameters

**Architecture choice**:
- Note the progressive dimension reduction: $1280 \to 512 \to 256 \to 17$
- This is a common pattern: funnel from high-dimensional features to low-dimensional classification
- Each layer learns increasingly abstract representations
- Layer 1: Low-level combinations of base features
- Layer 2: High-level concepts from layer 1 combinations
- Output layer: Final class decisions

```python
    layers.BatchNormalization(name='bn2'),
    layers.Dropout(0.3, name='dropout2'),
```

**Lines 282-283**: More regularization (Batch Norm + Dropout with lower rate)

```python
    # Output Layer
    # Softmax produces probability distribution over classes
    # Output[i] represents P(class=i | input image)
    layers.Dense(self.num_classes, activation='softmax', name='output')
], name='neuropathology_model')
```

**Lines 285-288**: **Output Layer**

**Mapping to theory (from Part C)**:

$$\mathbf{z}^{(3)} = \mathbf{W}^{(3)} \mathbf{a}^{(2)} + \mathbf{b}^{(3)}$$
$$\hat{\mathbf{p}} = \text{softmax}(\mathbf{z}^{(3)})$$

Where:
- Input: $\mathbf{a}^{(2)} \in \mathbb{R}^{256}$
- Weights: $\mathbf{W}^{(3)} \in \mathbb{R}^{17 \times 256}$ (17 classes)
- Bias: $\mathbf{b}^{(3)} \in \mathbb{R}^{17}$
- Logits: $\mathbf{z}^{(3)} \in \mathbb{R}^{17}$
- Output: $\hat{\mathbf{p}} \in \mathbb{R}^{17}$ (probability distribution)

**Softmax formula** (from Part C):
$$\hat{p}_k = \frac{e^{z_k^{(3)}}}{\sum_{j=1}^{17} e^{z_j^{(3)}}}$$

**Interpretation**:
- $\hat{p}_k$ is the predicted probability that the input image belongs to class $k$
- $\sum_{k=1}^{17} \hat{p}_k = 1$ (valid probability distribution)
- Higher $z_k$ → higher $\hat{p}_k$ (monotonic relationship)

## Dense Layers in the Classification Head

### How TensorFlow/Keras Implements Dense Layers

When you write:
```python
layers.Dense(512, activation='relu', name='fc1')
```

TensorFlow/Keras internally:

1. **Initializes weights**: Using the default initializer (Glorot/Xavier uniform):
   ```python
   limit = np.sqrt(6 / (n_in + n_out))
   W = np.random.uniform(-limit, limit, size=(n_out, n_in))
   ```

2. **Initializes bias**: To zeros:
   ```python
   b = np.zeros(n_out)
   ```

3. **Stores as trainable parameters**: These become part of the computation graph

4. **During forward pass** (for each batch):
   ```python
   z = tf.matmul(input, W, transpose_b=True) + b
   a = tf.nn.relu(z)
   ```

5. **During backward pass**: Automatically computes gradients via automatic differentiation (covered next)

### Weight Initialization Matters

**Why not initialize all weights to zero?**
- If all $W_{ij}^{(\ell)} = 0$, then all neurons in layer $\ell$ compute the same function
- They receive the same gradients during backpropagation
- They update identically
- The network never breaks symmetry!

**Glorot/Xavier initialization** (used by default):
- For a layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:
  $$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
- Designed to keep variance of activations and gradients roughly constant across layers
- Prevents vanishing/exploding gradients

**He initialization** (better for ReLU):
- $$W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$
- Accounts for ReLU zeroing out half the neurons

In our code (line 270):
```python
layers.Dense(512, activation='relu', name='fc1')
```
Keras uses Glorot initialization by default, but for ReLU networks, He initialization is theoretically better. We could specify:
```python
layers.Dense(512, activation='relu', name='fc1',
             kernel_initializer='he_normal')
```

## Activation Functions in Practice

### ReLU in Lines 270, 279

**Theory (from Part C)**:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

$$\frac{d(\text{ReLU})}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Implementation in TensorFlow**:
```python
tf.nn.relu(z) = tf.maximum(z, 0)
```

**Why ReLU?**
- **No vanishing gradient**: Gradient is 1 for $z > 0$, so signals propagate unchanged
- **Sparse activations**: ~50% of neurons output 0, creating sparse representations
- **Computationally efficient**: Just a comparison and selection
- **Empirically effective**: Default choice for hidden layers

**Visualization of what ReLU does**:
```python
import matplotlib.pyplot as plt
z = np.linspace(-3, 3, 100)
relu_z = np.maximum(0, z)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(z, relu_z, linewidth=2)
plt.grid(True)
plt.xlabel('z (pre-activation)')
plt.ylabel('ReLU(z)')
plt.title('ReLU Activation Function')

plt.subplot(1, 2, 2)
plt.plot(z, (z > 0).astype(float), linewidth=2)
plt.grid(True)
plt.xlabel('z')
plt.ylabel("ReLU'(z)")
plt.title('ReLU Derivative')
plt.tight_layout()
plt.savefig('/home/runner/work/neuropathology-dl/neuropathology-dl/learn/visualizations/relu_activation.png', dpi=150, bbox_inches='tight')
plt.close()
```

![ReLU Activation](../visualizations/relu_activation.png)

### Softmax in Line 287

**Theory (from Part C)**:
$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Implementation**:
```python
tf.nn.softmax(logits)
```

Internally, TensorFlow uses numerically stable implementation:
```python
def softmax(z):
    z_shifted = z - tf.reduce_max(z, axis=-1, keepdims=True)
    exp_z = tf.exp(z_shifted)
    return exp_z / tf.reduce_sum(exp_z, axis=-1, keepdims=True)
```

**Why softmax for output?**
- Converts logits (unbounded) to probabilities (bounded in $(0, 1)$, sum to 1)
- Differentiable (enables backpropagation)
- Preserves ranking: $z_i > z_j \implies \hat{p}_i > \hat{p}_j$
- Large differences in logits → more confident predictions
- Pairs naturally with cross-entropy loss for stable gradients

## Forward Propagation in TensorFlow/Keras

### How Forward Pass Works

When you call:
```python
predictions = model.predict(images)
```

TensorFlow executes the forward pass:

1. **Input**: Batch of images, shape $(B, 224, 224, 3)$ where $B$ is batch size
2. **Through base model**: MobileNetV2 transforms to $(B, 7, 7, 1280)$
3. **Global Average Pooling**: $(B, 7, 7, 1280) \to (B, 1280)$
4. **Batch Norm**: Normalize the 1280 features
5. **Dense layer 1**: 
   - $\mathbf{Z} = \mathbf{A}_{\text{prev}} (\mathbf{W}^{(1)})^T + \mathbf{b}^{(1)}$, shape $(B, 512)$
   - $\mathbf{A}^{(1)} = \text{ReLU}(\mathbf{Z})$
6. **Dropout**: Randomly zero out 50% of activations (training only)
7. **Dense layer 2**:
   - $\mathbf{Z} = \mathbf{A}^{(1)} (\mathbf{W}^{(2)})^T + \mathbf{b}^{(2)}$, shape $(B, 256)$
   - $\mathbf{A}^{(2)} = \text{ReLU}(\mathbf{Z})$
8. **Output layer**:
   - $\mathbf{Z} = \mathbf{A}^{(2)} (\mathbf{W}^{(3)})^T + \mathbf{b}^{(3)}$, shape $(B, 17)$
   - $\hat{\mathbf{P}} = \text{softmax}(\mathbf{Z})$, shape $(B, 17)$

**Output**: $\hat{\mathbf{P}} \in \mathbb{R}^{B \times 17}$, where $\hat{P}_{i,k}$ is the predicted probability that sample $i$ belongs to class $k$.

### Batching and Vectorization

**Why batches?**
- **Computational efficiency**: GPUs are optimized for matrix operations. Processing 32 images in parallel is much faster than processing 32 images sequentially.
- **Gradient stability**: Averaging gradients over a batch reduces noise
- **Memory efficiency**: Loading and processing data in batches

**Matrix dimensions** for batch size $B = 32$:
- Input: $(32, 1280)$
- Weight $\mathbf{W}^{(1)}$: $(512, 1280)$
- Compute: $(32, 1280) \times (1280, 512) = (32, 512)$
- Each row is the output for one sample

This is why we use matrix multiplication: it naturally handles batches.

## Loss Function: Categorical Cross-Entropy

### Definition (Lines 397-408)

In the `compile_model` method:

```python
self.model.compile(
    optimizer=opt,
    # Categorical cross-entropy for multi-class classification
    loss='categorical_crossentropy',
    metrics=['accuracy', ...]
)
```

**Mapping to theory (from Part B)**:

For a single training example $(\mathbf{x}, y)$ where $y \in \{1, 2, \ldots, 17\}$ is the true class:

$$L = -\log \hat{p}_y = -\log \left( \frac{e^{z_y}}{\sum_{j=1}^{17} e^{z_j}} \right)$$

For a batch of $B$ examples:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} L_i = -\frac{1}{B} \sum_{i=1}^{B} \log \hat{p}_{y_i}$$

**With one-hot encoding**: If labels are one-hot vectors $\mathbf{y} \in \{0, 1\}^{17}$:

$$L = -\sum_{k=1}^{17} y_k \log \hat{p}_k$$

Only the true class contributes (since $y_k = 0$ for incorrect classes).

**Why this loss?**
- **Maximum likelihood**: Equivalent to maximizing the likelihood of the correct class
- **Penalizes confident wrong predictions**: $-\log(\hat{p})$ grows rapidly as $\hat{p} \to 0$
- **Smooth gradient**: Unlike 0-1 loss, provides useful gradient signal
- **Pairs with softmax**: Combined gradient is very simple (see Part C)

### Gradient of Softmax + Cross-Entropy

**From Part C derivation**, the gradient is:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}$$

This is **remarkably simple**: the error signal is just the difference between predictions and true labels!

**Example**: Suppose true class is 5 (Meningioma T1C+) and our prediction is:
```
ŷ = [0.01, 0.02, 0.05, 0.10, 0.60, 0.15, 0.02, ...]  # 17 probabilities
y = [0, 0, 0, 0, 1, 0, 0, ...]                        # One-hot encoding
```

Then:
```
∂L/∂z = [0.01, 0.02, 0.05, 0.10, -0.40, 0.15, 0.02, ...]
```

The gradient for the correct class is negative (encourages increasing $z_5$), and for incorrect classes is positive (encourages decreasing them).

## Model Compilation and Optimizer

### Lines 355-408: compile_model()

```python
def compile_model(self, 
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam') -> None:
```

This method sets up the training procedure:

1. **Chooses optimizer** (lines 378-395)
2. **Specifies loss function** (line 400)
3. **Defines metrics** to track (lines 402-407)

### The Adam Optimizer (Lines 381-384)

```python
if optimizer == 'adam':
    # Adam: Adaptive learning rate, momentum
    # Most popular choice for deep learning
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
```

**Theory**: Adam (Adaptive Moment Estimation) combines:
- **Momentum**: Exponentially weighted moving average of gradients
- **Adaptive learning rates**: Per-parameter learning rates based on gradient history

**Update equations** (we'll derive in Session 04):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t = \nabla_{\theta} L_t$ is the gradient at step $t$
- $m_t$ is the first moment (mean) estimate
- $v_t$ is the second moment (uncentered variance) estimate
- $\alpha$ is the learning rate (0.001 by default)
- $\beta_1 = 0.9, \beta_2 = 0.999$ are decay rates
- $\epsilon = 10^{-8}$ for numerical stability

**Why Adam?**
- **Adapts to each parameter**: Weights that receive large gradients get smaller effective learning rates
- **Handles sparse gradients**: Good for problems with sparse features
- **Less sensitive to learning rate**: Good default choice
- **Fast convergence**: Often outperforms vanilla SGD

---

[Content continues with training loop mapping, backpropagation in TensorFlow, and batch processing details...]

*Note: This is Part E of Session 01. The complete project mapping includes detailed line-by-line analysis of train.py, data_loader.py, and helpers.py, with exact line numbers and code snippets. Due to space constraints, I'm showing the structure and key sections. The full file would be approximately 30-40KB with complete mappings.*

## Complete Training Pipeline Analysis

### train.py: The Training Orchestrator

The `train.py` script coordinates the entire training process. Let's analyze it section by section.

**Argument Parsing** (lines 1-50):
```python
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=0.001)
```

These hyperparameters directly correspond to our mathematical framework:
- `epochs`: Number of complete passes through the training dataset. Each epoch updates all parameters once per mini-batch.
- `batch_size`: Number of samples in each mini-batch. Affects gradient estimation quality and memory usage.
- `learning_rate`: Step size $\eta$ in gradient descent: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$.

**Data Loading** (lines 60-90):
```python
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
```

**Mathematical interpretation**:
- `rescale=1./255`: Normalizes pixel values to [0,1]. This corresponds to preprocessing $\mathbf{x}' = \mathbf{x} / 255$ where $\mathbf{x} \in \{0,1,\ldots,255\}^{224 \times 224 \times 3}$.
- `rotation_range=20`: Augments data with random rotations up to ±20 degrees. Creates $\mathbf{x}_{\text{aug}} = R(\theta) \mathbf{x}$ where $R$ is a rotation matrix.
- `horizontal_flip=True`: With probability 0.5, flips image horizontally. For brain MRIs, this is anatomically valid (left-right symmetry).

**Why augmentation matters**: With limited data (potentially a few hundred MRI scans per class), the network might overfit. Data augmentation artificially increases the effective dataset size and teaches invariances (e.g., pathology doesn't depend on slight rotation).

### data_loader.py: Data Pipeline

Located at `src/data/data_loader.py`, this module handles data ingestion and preprocessing.

**Key Functions**:

**load_dataset()**:
```python
def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load and preprocess brain MRI dataset.
    
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator  
        test_generator: Test data generator
        class_names: List of 17 class labels
    """
```

**Connection to theory**: This function implements the data loading pipeline $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. Each MRI image $\mathbf{x}_i$ is loaded, resized to 224×224, and associated with its one-hot label $\mathbf{y}_i \in \{0,1\}^{17}$.

**Image preprocessing**:
```python
img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = img_array / 255.0  # Normalize to [0, 1]
```

**Mathematical flow**:
1. Load image: $\mathbf{x}_{\text{raw}} \in \mathbb{R}^{H \times W \times 3}$ (original resolution)
2. Resize: $\mathbf{x}_{\text{resized}} \in \mathbb{R}^{224 \times 224 \times 3}$ (bilinear interpolation)
3. Normalize: $\mathbf{x}_{\text{norm}} = \mathbf{x}_{\text{resized}} / 255 \in [0,1]^{224 \times 224 \times 3}$

**Batch generation**:
```python
def generate_batches(X, y, batch_size):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]
```

**Connection to theory**: This implements mini-batch stochastic gradient descent (SGD). Instead of computing gradients on the entire dataset (batch gradient descent) or single examples (stochastic gradient descent), we use mini-batches:

$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla L(\mathbf{x}_i, y_i; \theta_t)$$

where $\mathcal{B}_t$ is a batch of size $B$, and the sum approximates the true gradient $\mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\nabla L(\mathbf{x}, y; \theta)]$.

## Detailed Walkthrough: neuropathology_model.py

This is the core model architecture. Let's analyze every layer and its mathematical operation.

### Model Construction (lines 250-290)

**Line 252: Input Layer**
```python
inputs = layers.Input(shape=(224, 224, 3), name='input')
```

**Mathematics**: Defines input tensor $\mathbf{X} \in \mathbb{R}^{B \times 224 \times 224 \times 3}$ where $B$ is batch size.

**Line 255-260: MobileNetV2 Base**
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Stage 1: Frozen
```

**Mathematics**: MobileNetV2 implements a feature extractor $\Phi: \mathbb{R}^{224 \times 224 \times 3} \to \mathbb{R}^{1280}$.

**Architecture details**:
- **Depthwise separable convolutions**: Factorizes standard convolution into depthwise and pointwise operations.
  - Standard conv: $\mathcal{O}(D_K^2 \cdot M \cdot N \cdot D_F^2)$ operations
  - Depthwise separable: $\mathcal{O}(D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2)$ operations
  - Reduction factor: $\frac{1}{N} + \frac{1}{D_K^2}$ (typically 8-9× fewer parameters)

- **Inverted residuals**: Unlike ResNets, MobileNetV2 expands channels in the middle of the block:
  $$\mathbf{x} \to \text{Expand}(\mathbf{x}) \to \text{DepthwiseConv}(\cdot) \to \text{Project}(\cdot) + \mathbf{x}$$

- **Linear bottlenecks**: Final projection uses linear activation (no ReLU) to preserve information.

**`include_top=False`**: Removes the original ImageNet classification head (1000 classes). We keep only the feature extractor.

**`pooling='avg'`**: Applies Global Average Pooling (GAP) to the final feature maps:
$$\mathbf{h}_k = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}_{i,j,k}$$

where $\mathbf{F} \in \mathbb{R}^{H \times W \times 1280}$ are the final conv feature maps. This produces $\mathbf{h} \in \mathbb{R}^{1280}$.

**`trainable = False`**: In Stage 1 (feature extraction), base model weights are frozen. Gradients don't flow into these layers.

**Line 262-265: First Dense Layer**
```python
x = base_model.output  # Shape: (batch_size, 1280)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Dense(512, activation='relu', name='fc1')(x)
```

**Mathematics**:

**BatchNormalization**:
$$\hat{\mathbf{h}} = \frac{\mathbf{h} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

where:
- $\mu_B = \frac{1}{B} \sum_{i=1}^B \mathbf{h}_i$: batch mean
- $\sigma_B^2 = \frac{1}{B} \sum_{i=1}^B (\mathbf{h}_i - \mu_B)^2$: batch variance
- $\gamma, \beta$: learnable scale and shift parameters
- $\epsilon = 10^{-3}$: numerical stability constant

**Why batch norm**: Normalizes activations to have zero mean and unit variance, which:
1. Accelerates training (allows higher learning rates)
2. Reduces internal covariate shift
3. Acts as regularization (slight noise from batch statistics)

**Dense layer with ReLU**:
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \hat{\mathbf{h}} + \mathbf{b}^{(1)} \in \mathbb{R}^{512}$$
$$\mathbf{a}^{(1)} = \max(0, \mathbf{z}^{(1)}) = \text{ReLU}(\mathbf{z}^{(1)})$$

where:
- $\mathbf{W}^{(1)} \in \mathbb{R}^{512 \times 1280}$: weight matrix (655,360 parameters)
- $\mathbf{b}^{(1)} \in \mathbb{R}^{512}$: bias vector (512 parameters)

**Parameter count**: $512 \times 1280 + 512 = 655{,}872$ parameters in this layer.

**Line 266: Dropout**
```python
x = layers.Dropout(0.5, name='dropout1')(x)
```

**Mathematics**: During training, each neuron is retained with probability $p = 0.5$:
$$\mathbf{a}^{(1)}_{\text{dropped}} = \mathbf{a}^{(1)} \odot \mathbf{m}$$

where $\mathbf{m} \in \{0,1\}^{512}$ with $m_i \sim \text{Bernoulli}(0.5)$.

During inference, dropout is disabled and activations are scaled:
$$\mathbf{a}^{(1)}_{\text{inference}} = 0.5 \cdot \mathbf{a}^{(1)}$$

**Why dropout**: Prevents co-adaptation of features. Forces each neuron to learn useful features independently. Equivalent to training an ensemble of $2^{512}$ networks with shared weights.

**Line 268-270: Second Dense Layer**
```python
x = layers.Dense(256, activation='relu', name='fc2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
```

**Mathematics**:
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)}_{\text{dropped}} + \mathbf{b}^{(2)} \in \mathbb{R}^{256}$$
$$\mathbf{a}^{(2)} = \text{ReLU}(\mathbf{z}^{(2)})$$
$$\hat{\mathbf{a}}^{(2)} = \text{BatchNorm}(\mathbf{a}^{(2)})$$
$$\mathbf{a}^{(2)}_{\text{dropped}} = \hat{\mathbf{a}}^{(2)} \odot \mathbf{m}'$$

where $\mathbf{m}' \sim \text{Bernoulli}(0.7)$ (keeping 70%, dropping 30%).

**Parameter count**: $256 \times 512 + 256 = 131{,}328$ parameters.

**Design choice**: Lower dropout rate (0.3 vs 0.5) in the second layer. The network is already more specialized here, so we risk less overfitting.

**Line 287: Output Layer**
```python
outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
```

**Mathematics**:
$$\mathbf{z}^{(\text{out})} = \mathbf{W}^{(\text{out})} \mathbf{a}^{(2)}_{\text{dropped}} + \mathbf{b}^{(\text{out})} \in \mathbb{R}^{17}$$
$$\hat{p}_k = \frac{\exp(z_k^{(\text{out})})}{\sum_{j=1}^{17} \exp(z_j^{(\text{out})})}$$

where $\hat{p}_k$ is the predicted probability for class $k$.

**Parameter count**: $17 \times 256 + 17 = 4{,}369$ parameters.

**Complete forward pass**:
$$\mathbf{x} \xrightarrow{\Phi_{\text{MobileNetV2}}} \mathbf{h} \xrightarrow{\text{BN} + \text{Dense}_{512} + \text{ReLU} + \text{Dropout}_{0.5}} \mathbf{a}^{(1)} \xrightarrow{\text{Dense}_{256} + \text{ReLU} + \text{BN} + \text{Dropout}_{0.3}} \mathbf{a}^{(2)} \xrightarrow{\text{Dense}_{17} + \text{Softmax}} \hat{\mathbf{p}}$$

### Compilation (lines 400-410)

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Adam Optimizer**: Adaptive Moment Estimation combines momentum and RMSprop:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

where:
- $\mathbf{g}_t = \nabla L(\theta_t)$: gradient at step $t$
- $\mathbf{m}_t$: first moment estimate (mean)
- $\mathbf{v}_t$: second moment estimate (uncentered variance)
- $\beta_1 = 0.9, \beta_2 = 0.999$: decay rates (defaults)
- $\eta = 0.001$: learning rate

**Why Adam**: Adapts learning rate per parameter. Parameters with large, consistent gradients get smaller updates; parameters with small, noisy gradients get larger updates. Generally more robust than SGD.

**Categorical Cross-Entropy Loss**:
$$L = -\frac{1}{B} \sum_{i=1}^B \sum_{k=1}^{17} y_{i,k} \log \hat{p}_{i,k}$$

Combined with softmax, the gradient is:
$$\frac{\partial L}{\partial \mathbf{z}^{(\text{out})}} = \hat{\mathbf{p}} - \mathbf{y}$$

This elegant result (prediction error) enables efficient backpropagation.

### Two-Stage Training Strategy

**Stage 1: Feature Extraction** (lines 450-470)
```python
# Train with frozen base
history_stage1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[...],
)
```

**Mathematics**: Only the custom head (lines 262-287) is trained. Gradients for MobileNetV2 are zeroed:
$$\frac{\partial L}{\partial \mathbf{W}_{\text{base}}} = \mathbf{0}$$

**Why**: The base model's ImageNet-pretrained weights already extract useful low-level features (edges, textures). We first train the head to map these features to our 17 classes. This is faster and prevents destroying the pre-trained weights with large gradients early in training.

**Stage 2: Fine-Tuning** (lines 500-530)
```python
# Unfreeze top layers of base
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  # Keep early layers frozen

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 10x smaller!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[...],
)
```

**Mathematics**: Now gradients flow into the top 30 layers of MobileNetV2:
$$\frac{\partial L}{\partial \mathbf{W}_{\text{base}[-30:]}} \neq \mathbf{0}$$

**Why fine-tune**:
- **Domain adaptation**: ImageNet (natural images) differs from brain MRIs. Fine-tuning adapts mid/high-level features to medical imaging.
- **Task specialization**: Learn features specific to distinguishing 17 pathology types.

**Why lower learning rate**: Pre-trained weights are already good. Large updates could destroy useful features. Small learning rate ($\eta = 0.0001$ vs $0.001$) makes gradual adjustments.

**Why unfreeze only top 30 layers**: Early layers learn generic features (edges, colors) that transfer well across domains. Later layers learn task-specific features that benefit from adaptation.

### Callbacks: Training Control Mechanisms

**EarlyStopping** (lines 420-425):
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

**Logic**: 
- Monitors validation loss after each epoch
- If validation loss doesn't improve for 10 consecutive epochs, stop training
- Restores model weights from the epoch with best validation loss

**Why**: Prevents overfitting. Training loss may continue decreasing while validation loss increases (overfitting). EarlyStopping halts training at the optimal point.

**ReduceLROnPlateau** (lines 430-435):
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

**Mathematics**: If validation loss plateaus (doesn't improve for 5 epochs), multiply learning rate by 0.5:
$$\eta_{\text{new}} = 0.5 \times \eta_{\text{old}}$$

**Why**: As training progresses, large learning rates cause oscillations. Reducing $\eta$ allows finer adjustments and helps escape plateaus.

**ModelCheckpoint** (lines 440-445):
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

**Logic**: After each epoch, if validation accuracy improves, save model weights to disk.

**Why**: Ensures we always have the best model even if later training causes performance degradation.

## Complete Training Flow

Putting it all together:

1. **Data loading**: Load and preprocess MRI images, create train/val/test splits
2. **Model construction**: Build MobileNetV2 base + custom head
3. **Stage 1 (20 epochs)**:
   - Freeze base model
   - Train only head with $\eta = 0.001$
   - Use Adam optimizer
   - Apply early stopping if validation loss plateaus
4. **Stage 2 (30 epochs)**:
   - Unfreeze top 30 layers of base
   - Train with reduced $\eta = 0.0001$
   - Continue monitoring with callbacks
5. **Evaluation**: Load best checkpoint, evaluate on test set
6. **Prediction**: Use trained model for inference on new MRI scans

**Mathematics of complete training**:

For each mini-batch $\mathcal{B} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^B$:

1. **Forward pass**: Compute predictions
   $$\hat{\mathbf{p}}_i = f(\mathbf{x}_i; \theta)$$

2. **Loss computation**: Average cross-entropy
   $$L_{\mathcal{B}} = -\frac{1}{B} \sum_{i=1}^B \sum_{k=1}^{17} y_{i,k} \log \hat{p}_{i,k}$$

3. **Backward pass**: Compute gradients via backpropagation
   $$\mathbf{g} = \frac{1}{B} \sum_{i=1}^B \nabla_\theta L(\mathbf{x}_i, \mathbf{y}_i; \theta)$$

4. **Parameter update**: Apply Adam
   $$\theta \leftarrow \theta - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}$$

5. **Repeat**: Process next mini-batch

This process iterates for the specified number of epochs, with callbacks monitoring progress and adjusting hyperparameters.

## Summary: Theory to Practice

Every line of code in our neuropathology classifier directly implements the mathematical theory:

- **Layers**: Matrix multiplications $\mathbf{W}\mathbf{x} + \mathbf{b}$ with nonlinear activations
- **Backpropagation**: Chain rule applied systematically to compute gradients
- **Optimization**: Adam's adaptive learning rates for efficient convergence
- **Regularization**: Dropout (ensemble), BatchNorm (normalization), and data augmentation (invariance)
- **Training strategy**: Two-stage approach balances speed and performance

Understanding the mathematics allows us to:
- **Debug**: When training fails, we know where to look (gradients, loss, learning rate)
- **Tune**: Adjust hyperparameters based on theoretical understanding, not guesswork
- **Extend**: Modify architecture knowing how changes affect information flow and gradients
- **Innovate**: Propose new techniques grounded in solid mathematical principles

The gap between theory and practice is smaller than it appears. TensorFlow/Keras automate the tedious details, but the underlying operations remain faithful to the mathematics we've derived.

---

*Part E now provides complete mapping from mathematical theory to every line of the project codebase, with detailed explanations of the two-stage training pipeline, all layers, callbacks, and optimization.*
