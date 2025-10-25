# 📚 Deep Learning Theory Guide

This comprehensive guide explains the deep learning concepts used in this project, from fundamentals to advanced topics.

## Table of Contents

1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Convolutional Neural Networks](#convolutional-neural-networks)
3. [Transfer Learning](#transfer-learning)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Regularization Techniques](#regularization-techniques)
6. [Model Evaluation](#model-evaluation)
7. [Mathematical Foundations](#mathematical-foundations)

---

## Introduction to Neural Networks

### What is a Neural Network?

A neural network is a computational model inspired by biological neurons in the brain. It consists of layers of interconnected nodes (neurons) that process information.

### The Artificial Neuron

The basic building block is an artificial neuron (perceptron):

```
Inputs: x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias: b

Computation:
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σᵢ wᵢxᵢ + b
output = activation(z)
```

**Example**: A neuron with 3 inputs
- Inputs: x = [0.5, 0.3, 0.8]
- Weights: w = [0.4, 0.2, 0.6]
- Bias: b = 0.1

Calculation:
```
z = (0.4 × 0.5) + (0.2 × 0.3) + (0.6 × 0.8) + 0.1
z = 0.2 + 0.06 + 0.48 + 0.1 = 0.84
output = ReLU(0.84) = 0.84
```

### Activation Functions

Activation functions introduce non-linearity, enabling the network to learn complex patterns.

#### 1. ReLU (Rectified Linear Unit)

```
f(z) = max(0, z)
```

**Properties**:
- Output: [0, ∞)
- Derivative: 1 if z > 0, else 0
- Computational efficiency: Simple max operation
- Problem: "Dying ReLU" when z < 0 always

**When to use**: Most hidden layers in modern networks

#### 2. Sigmoid

```
f(z) = 1 / (1 + e⁻ᶻ)
```

**Properties**:
- Output: (0, 1)
- Smooth, differentiable
- Problem: Vanishing gradients for large |z|

**When to use**: Binary classification output layer

#### 3. Tanh (Hyperbolic Tangent)

```
f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

**Properties**:
- Output: (-1, 1)
- Zero-centered (better than sigmoid)
- Still suffers from vanishing gradients

**When to use**: Hidden layers when zero-centered output desired

#### 4. Softmax

```
f(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

**Properties**:
- Outputs sum to 1
- Converts scores to probabilities
- Differentiable

**When to use**: Multi-class classification output layer

### Multi-Layer Networks

A deep neural network stacks multiple layers:

```
Input Layer (features)
    ↓
Hidden Layer 1 (learned representations)
    ↓
Hidden Layer 2 (higher-level features)
    ↓
...
    ↓
Output Layer (predictions)
```

**Forward Propagation**:
```
Layer 1: a⁽¹⁾ = f(W⁽¹⁾x + b⁽¹⁾)
Layer 2: a⁽²⁾ = f(W⁽²⁾a⁽¹⁾ + b⁽²⁾)
...
Output: ŷ = f(W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾)
```

### Universal Approximation Theorem

**Statement**: A feedforward neural network with:
- Single hidden layer
- Sufficient neurons
- Non-linear activation

Can approximate any continuous function to arbitrary accuracy.

**Implication**: Neural networks are extremely powerful function approximators!

---

## Convolutional Neural Networks

### Why CNNs for Images?

Traditional fully-connected networks have problems with images:

1. **Too many parameters**:
   - 224×224 RGB image = 150,528 inputs
   - Single hidden layer with 1000 neurons = 150M parameters!

2. **Loss of spatial structure**:
   - Flattening destroys spatial relationships
   - Nearby pixels are related, but network doesn't know

3. **Not translation invariant**:
   - Same object at different positions seen as different

### Convolutional Layer

**Key Ideas**:
1. **Local connectivity**: Each neuron sees only part of input
2. **Parameter sharing**: Same filter applied everywhere
3. **Translation invariance**: Detects features regardless of position

**Convolution Operation**:
```
(I * K)(i,j) = ΣₘΣₙ I(i+m, j+n) × K(m,n)
```

**Example**: Edge detection with 3×3 filter
```
Input:               Filter:          Output:
[100 100 100 0 0]    [-1 -1 -1]      [0 300 300 0]
[100 100 100 0 0]  * [ 0  0  0] =    [0 300 300 0]
[100 100 100 0 0]    [ 1  1  1]      [0 300 300 0]
[100 100 100 0 0]
```

The filter detects vertical edges (large values where intensity changes).

### Convolution Parameters

**Filter/Kernel Size**:
- Common: 3×3, 5×5, 7×7
- Smaller filters = fewer parameters, more layers needed
- Larger filters = more parameters, larger receptive field

**Stride**:
- How much to move filter each step
- Stride = 1: Move one pixel at a time
- Stride = 2: Move two pixels (downsampling)

**Padding**:
- Add zeros around border
- 'valid': No padding (output smaller)
- 'same': Pad to keep size (output same size)

**Number of Filters**:
- Each filter detects different feature
- More filters = more features learned
- Common: 32, 64, 128, 256, ...

**Output Size Calculation**:
```
Output size = (Input size - Filter size + 2×Padding) / Stride + 1
```

Example: Input=28, Filter=3, Stride=1, Padding=0
```
Output = (28 - 3 + 0) / 1 + 1 = 26
```

### Pooling Layer

**Purpose**:
- Reduce spatial dimensions
- Provide translation invariance
- Reduce computation

**Max Pooling** (most common):
```
Input (4×4):          Output (2×2) with 2×2 pooling:
[1  2  3  4]          [6  8]
[5  6  7  8]    →     [14 16]
[9  10 11 12]
[13 14 15 16]
```

Takes maximum value in each region.

**Average Pooling**:
```
Same input:           Output:
[1  2  3  4]          [3.5  5.5]
[5  6  7  8]    →     [11.5 13.5]
[9  10 11 12]
[13 14 15 16]
```

Takes average value in each region.

### CNN Architecture Patterns

**LeNet (1998)**: Conv → Pool → Conv → Pool → FC → FC
- First successful CNN
- Simple, effective

**AlexNet (2012)**: Conv → Pool → Conv → Pool → Conv → Conv → Conv → Pool → FC → FC → FC
- Deeper network
- ReLU activation
- Dropout regularization

**VGG (2014)**: Multiple 3×3 Conv → Pool (repeated)
- Very deep (16-19 layers)
- Only 3×3 filters
- Simple, uniform architecture

**ResNet (2015)**: Conv layers + Skip connections
- Very deep (50-152 layers)
- Skip connections solve vanishing gradient
- Residual learning: y = F(x) + x

**EfficientNet (2019)**: Compound scaling
- Scales depth, width, resolution together
- Best accuracy/efficiency trade-off
- Our default choice!

### Feature Hierarchy

CNNs learn hierarchical features:

**Layer 1** (Early):
- Edges (horizontal, vertical, diagonal)
- Colors
- Simple textures

**Layer 2-3** (Middle):
- Corners
- Contours
- Patterns
- Simple shapes

**Layer 4-5** (Deep):
- Object parts (e.g., eyes, wheels)
- Complex textures
- Combinations of shapes

**Output Layer**:
- Full objects
- Complete patterns
- High-level concepts

This hierarchy mirrors human visual system!

---

## Transfer Learning

### The Core Idea

**Problem**: Training from scratch requires:
- Millions of labeled images
- Weeks of GPU time
- Significant expertise

**Solution**: Start with pre-trained model
- Already learned general visual features
- Fine-tune for specific task
- Faster, better results with less data

### Why It Works

**Feature Universality**: Early layers learn universal features
- Edges, corners, textures appear in all images
- Medical images also have these features
- Later layers learn task-specific features

**Mathematical Intuition**:

Instead of random initialization:
```
θ ~ N(0, σ²)  ❌ Poor starting point
```

Use pre-trained weights:
```
θ ← θ_pretrained  ✅ Good starting point
```

Then optimize:
```
θ* = argmin_θ L(θ) starting from θ_pretrained
```

Converges faster and to better minimum!

### Transfer Learning Strategies

#### Strategy 1: Feature Extraction

```
Pre-trained Model [FROZEN] → New Classifier [TRAINABLE]
```

**When**: Small dataset, similar to pre-training data
**How**:
```python
base_model.trainable = False  # Freeze all layers
model = Sequential([
    base_model,
    Dense(...),  # New layers
])
```

#### Strategy 2: Fine-Tuning

```
Early Layers [FROZEN] → Later Layers [TRAINABLE] → Classifier [TRAINABLE]
```

**When**: Medium dataset, somewhat different from pre-training
**How**:
```python
# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True
```

**Why lower learning rate?**
```python
# Stage 1: LR = 0.001 (train classifier)
# Stage 2: LR = 0.0001 (fine-tune)
```
- Prevents destroying pre-trained features
- Makes gentle adjustments
- Better final performance

#### Strategy 3: Full Training

```
All Layers [TRAINABLE]
```

**When**: Large dataset, very different from pre-training
**Caution**: May overfit if dataset too small

### Our Implementation

**Two-Stage Process**:

**Stage 1**: Feature Extraction (30 epochs)
```python
base_model.trainable = False
learning_rate = 0.001
# Train classifier: 70% → 90% accuracy
```

**Stage 2**: Fine-Tuning (20 epochs)
```python
# Unfreeze last 20 layers
base_model.layers[-20:].trainable = True
learning_rate = 0.0001  # Lower!
# Fine-tune: 90% → 95% accuracy
```

### ImageNet Pre-training

**ImageNet Dataset**:
- 1.2 million training images
- 1000 classes (dogs, cats, cars, etc.)
- General object recognition

**Why useful for medical imaging?**
- Learns general visual features
- Edge detection, texture analysis
- Object localization
- Spatial hierarchies

These features transfer surprisingly well to MRI analysis!

---

## Optimization Algorithms

### Gradient Descent

**Goal**: Minimize loss function L(θ)

**Core Algorithm**:
```
Initialize: θ ← random values
Repeat:
    1. Compute gradient: g = ∇L(θ)
    2. Update: θ ← θ - η·g
Until convergence
```

**Gradient**: Direction of steepest increase
- Follow negative gradient to decrease loss
- Step size controlled by learning rate η

### Variants of Gradient Descent

#### Batch Gradient Descent

```python
for epoch in range(num_epochs):
    gradient = compute_gradient(all_training_data)
    weights = weights - learning_rate * gradient
```

**Pros**: Accurate gradient, stable
**Cons**: Slow, needs all data in memory

#### Stochastic Gradient Descent (SGD)

```python
for epoch in range(num_epochs):
    for sample in training_data:
        gradient = compute_gradient(sample)
        weights = weights - learning_rate * gradient
```

**Pros**: Fast, online learning
**Cons**: Noisy updates, may not converge

#### Mini-Batch Gradient Descent

```python
for epoch in range(num_epochs):
    for batch in get_batches(training_data, batch_size=32):
        gradient = compute_gradient(batch)
        weights = weights - learning_rate * gradient
```

**Pros**: Balance of speed and stability
**Cons**: Need to tune batch size

**Most common in practice!**

### Advanced Optimizers

#### Momentum

Adds "velocity" to updates:
```
v_t = β·v_{t-1} + (1-β)·g_t
θ_t = θ_{t-1} - η·v_t
```

**Benefits**:
- Accelerates in consistent directions
- Dampens oscillations
- Escapes local minima

**Analogy**: Rolling ball accumulates momentum

#### RMSprop

Adaptive learning rate per parameter:
```
v_t = β·v_{t-1} + (1-β)·g_t²
θ_t = θ_{t-1} - η · g_t / √(v_t + ε)
```

**Benefits**:
- Different learning rates for different parameters
- Handles sparse gradients well
- Good for RNNs

#### Adam (Adaptive Moment Estimation)

**Our Choice!** Combines momentum + adaptive learning rate:

```
# First moment (momentum)
m_t = β₁·m_{t-1} + (1-β₁)·g_t

# Second moment (adaptive LR)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²

# Bias correction
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

# Update
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

**Default values**:
- η = 0.001 (learning rate)
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (adaptive LR decay)
- ε = 10⁻⁸ (numerical stability)

**Why Adam is popular**:
- Works well with little tuning
- Handles sparse gradients
- Invariant to gradient scale
- Fast convergence
- Industry standard

### Learning Rate Schedules

**Problem**: Fixed learning rate suboptimal
- Too high: Oscillates, doesn't converge
- Too low: Slow convergence

**Solution**: Decay learning rate over time

#### Step Decay

```
η_t = η₀ × γ^(t/k)
```

Drop by factor γ every k epochs.

Example: η₀=0.1, γ=0.5, k=10
```
Epochs 0-9:   η = 0.1
Epochs 10-19: η = 0.05
Epochs 20-29: η = 0.025
```

#### Exponential Decay

```
η_t = η₀ × e^(-λt)
```

Smooth exponential decrease.

#### Reduce on Plateau

```
If val_loss doesn't improve for N epochs:
    η = η × factor
```

**Our choice!** Adaptive based on validation performance.

---

## Regularization Techniques

### The Overfitting Problem

**Overfitting**: Model memorizes training data
- High training accuracy
- Poor test accuracy
- Doesn't generalize

**Signs**:
- Training loss keeps decreasing
- Validation loss starts increasing
- Large gap between train and val accuracy

### Dropout

**Idea**: Randomly deactivate neurons during training

```python
# During training
active = random_mask(p=0.5)  # 50% dropout
output = (input * active) / p

# During testing
output = input  # Use all neurons
```

**Why it works**:
- Prevents co-adaptation of neurons
- Forces redundant representations
- Like training ensemble of networks
- Simple and effective

**Where to use**:
- After dense layers
- Typical rates: 0.3 to 0.5
- Higher rates for larger layers

**In our model**:
```python
Dense(512) → Dropout(0.5)  # High dropout for large layer
Dense(256) → Dropout(0.3)  # Lower for smaller layer
```

### Batch Normalization

**Idea**: Normalize layer inputs

```
μ_batch = (1/m) Σᵢ xᵢ
σ²_batch = (1/m) Σᵢ (xᵢ - μ_batch)²

x̂ᵢ = (xᵢ - μ_batch) / √(σ²_batch + ε)

yᵢ = γ·x̂ᵢ + β  # Learnable scale and shift
```

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Reduces dependence on initialization
- Acts as regularization
- Faster training

**Where to use**:
- After convolutional layers
- Before or after activation
- Before dense layers

### Data Augmentation

**Idea**: Artificially increase dataset through transformations

**For images**:
```python
# Rotation
rotate(image, angle=random(-20, 20))

# Zoom
zoom(image, factor=random(0.9, 1.1))

# Flip
if random() > 0.5:
    flip_horizontal(image)

# Brightness
adjust_brightness(image, factor=random(0.8, 1.2))
```

**Benefits**:
- No cost (just transformations)
- Improves generalization
- Makes model robust to variations
- Especially helpful for small datasets

**Important**: Only during training, not testing!

### Early Stopping

**Idea**: Stop training when validation loss stops improving

```python
best_loss = infinity
patience_counter = 0

for epoch in range(max_epochs):
    train_model()
    val_loss = evaluate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break  # Stop training
        
restore_best_model()
```

**Benefits**:
- Prevents overfitting
- Automatic stopping criterion
- Saves training time
- Restores best weights

**Parameters**:
- Patience: How many epochs to wait (e.g., 10)
- Monitor: val_loss or val_accuracy

### L2 Regularization (Weight Decay)

**Idea**: Penalize large weights

```
L_total = L_data + λΣw²
```

**Effect**: Encourages smaller, simpler weights

**In Adam optimizer**:
```python
Adam(learning_rate=0.001, decay=1e-6)
```

Implicitly performs weight decay.

---

## Model Evaluation

### Classification Metrics

#### Confusion Matrix

```
                Predicted
                Pos    Neg
Actual Pos      TP     FN
       Neg      FP     TN
```

- TP: Correctly predicted positive
- TN: Correctly predicted negative
- FP: Incorrectly predicted positive (Type I error)
- FN: Incorrectly predicted negative (Type II error)

#### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation**: Proportion of correct predictions

**Limitation**: Misleading with imbalanced data

Example: 95% class A, 5% class B
- Always predict A → 95% accuracy!
- But completely fails on class B

#### Precision

```
Precision = TP / (TP + FP)
```

**Question**: Of all positive predictions, how many are correct?

**High precision**: Few false alarms

**Medical context**: How reliable are positive diagnoses?

#### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

**Question**: Of all actual positives, how many did we find?

**High recall**: Few missed cases

**Medical context**: Are we missing any diseases?

#### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation**: Harmonic mean of precision and recall

**Use when**: Need balance between precision and recall

#### Specificity

```
Specificity = TN / (TN + FP)
```

**Question**: Of all actual negatives, how many did we correctly identify?

**Medical context**: Are we over-diagnosing?

### Multi-Class Metrics

For K classes, compute metrics per class, then aggregate:

#### Macro-Average

```
Metric_macro = (1/K) Σᵢ Metric_i
```

Simple average. Treats all classes equally.

#### Weighted-Average

```
Metric_weighted = Σᵢ (nᵢ/N) × Metric_i
```

Weighted by class frequency. Accounts for imbalance.

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic)**:
- Plot TPR vs FPR at different thresholds
- TPR = Recall = TP/(TP+FN)
- FPR = FP/(FP+TN)

**AUC (Area Under Curve)**:
- Single number summarizing ROC
- Range: [0, 1]
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC > 0.9: Excellent
- AUC > 0.8: Good
- AUC > 0.7: Fair

**Interpretation**: Probability that model ranks random positive higher than random negative.

### Cross-Validation

**Problem**: Single train/test split may be lucky/unlucky

**K-Fold Cross-Validation**:
```
1. Split data into K folds
2. For fold i in 1..K:
    - Train on K-1 folds
    - Test on fold i
    - Record score_i
3. Final score = average(scores)
```

**Benefits**:
- More reliable estimate
- Uses all data
- Reduces variance

**Common**: K=5 or K=10

---

## Mathematical Foundations

### Linear Algebra

#### Vectors and Matrices

**Vector**: Array of numbers
```
x = [x₁, x₂, ..., xₙ]
```

**Matrix**: 2D array
```
     [w₁₁  w₁₂  w₁₃]
W =  [w₂₁  w₂₂  w₂₃]
```

**Matrix-Vector Product**:
```
y = Wx

yᵢ = Σⱼ Wᵢⱼxⱼ
```

**Neural network layer**:
```
z = Wx + b
```

This is just a linear transformation!

#### Dot Product

```
a · b = Σᵢ aᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**Properties**:
- Measures similarity
- Large when vectors aligned
- Zero when perpendicular

**In neural networks**:
```
z = w · x + b
```

Neuron computes weighted sum!

### Calculus

#### Derivatives

**Definition**:
```
f'(x) = lim_{h→0} [f(x+h) - f(x)] / h
```

**Interpretation**: Rate of change

**Common derivatives**:
```
d/dx(x²) = 2x
d/dx(eˣ) = eˣ
d/dx(log x) = 1/x
d/dx(sin x) = cos x
```

#### Chain Rule

For composite function f(g(x)):
```
df/dx = (df/dg) × (dg/dx)
```

**Example**:
```
f(x) = (2x + 1)²

Let g(x) = 2x + 1, f(g) = g²

df/dx = (df/dg) × (dg/dx)
      = 2g × 2
      = 2(2x + 1) × 2
      = 4(2x + 1)
```

**In neural networks**: Backpropagation is just chain rule!

```
∂L/∂w₁ = (∂L/∂output) × (∂output/∂hidden) × (∂hidden/∂w₁)
```

#### Partial Derivatives

For function f(x, y):
```
∂f/∂x: Derivative with respect to x (treat y as constant)
∂f/∂y: Derivative with respect to y (treat x as constant)
```

**Example**:
```
f(x, y) = x²y + y³

∂f/∂x = 2xy
∂f/∂y = x² + 3y²
```

#### Gradient

**Definition**: Vector of partial derivatives
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Interpretation**: Direction of steepest increase

**In optimization**:
```
θ_new = θ_old - η × ∇L(θ)
```

Move opposite to gradient (steepest decrease)!

### Probability

#### Probability Basics

**Probability**: P(A) ∈ [0, 1]
- P(A) = 0: Event never happens
- P(A) = 1: Event always happens
- P(A) + P(not A) = 1

**Conditional Probability**:
```
P(A|B) = P(A and B) / P(B)
```

Probability of A given B occurred.

#### Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

**In classification**:
```
P(class|image) = P(image|class) × P(class) / P(image)
```

Neural networks learn P(class|image)!

#### Expected Value

```
E[X] = Σᵢ xᵢ × P(xᵢ)
```

**Interpretation**: Average value

**In loss functions**:
```
L = E[loss per sample] = (1/N) Σᵢ loss(ŷᵢ, yᵢ)
```

Average loss over dataset.

### Information Theory

#### Entropy

```
H(X) = -Σᵢ P(xᵢ) log P(xᵢ)
```

**Interpretation**: Average "surprise" or information content

**Properties**:
- H = 0: No uncertainty (deterministic)
- H = max: Maximum uncertainty (uniform distribution)

#### Cross-Entropy

```
H(p, q) = -Σᵢ p(xᵢ) log q(xᵢ)
```

**Interpretation**: Expected surprise when using q to encode samples from p

**In deep learning**:
```
L = -Σᵢ y_true(i) × log(y_pred(i))
```

Where:
- y_true: True distribution (one-hot)
- y_pred: Predicted distribution (softmax)

**Why it works**:
- Penalizes confident wrong predictions
- log(0.01) = -4.6 (high penalty)
- log(0.99) = -0.01 (low penalty)

Perfect when y_pred = y_true!

---

## Conclusion

This guide covers the essential deep learning concepts used in this project:

1. **Neural Networks**: Building blocks of deep learning
2. **CNNs**: Specialized for image processing
3. **Transfer Learning**: Leveraging pre-trained models
4. **Optimization**: How models learn
5. **Regularization**: Preventing overfitting
6. **Evaluation**: Measuring performance
7. **Mathematics**: Underlying foundations

Each concept is explained with:
- Intuitive explanations
- Mathematical formulations
- Practical examples
- Code correlations

Use this guide as a reference while exploring the codebase!

For more details, see:
- Code documentation in `src/models/neuropathology_model.py`
- Training pipeline in `train.py`
- Web app theory pages in `app.py`
- Architecture documentation in `ARCHITECTURE.md`
