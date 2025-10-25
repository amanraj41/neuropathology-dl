"""
Neural Network Model for Neuropathology Detection

This module implements the deep learning model architecture for detecting
neuropathological conditions from MRI images.

=== DEEP LEARNING FUNDAMENTALS ===

1. NEURAL NETWORKS - The Foundation
   
   A neural network is a computational model inspired by biological neurons.
   
   Mathematical Model of a Neuron:
   - Input: x = [x₁, x₂, ..., xₙ] (feature vector)
   - Weights: w = [w₁, w₂, ..., wₙ] (learned parameters)
   - Bias: b (learned offset parameter)
   - Activation: a = f(z) where z = w·x + b = Σ(wᵢxᵢ) + b
   
   The activation function f introduces non-linearity:
   - Without non-linearity, stacking layers just creates a linear function
   - Non-linearity allows learning complex patterns
   
   Common Activation Functions:
   - ReLU: f(z) = max(0, z)
     * Pros: Fast, no vanishing gradient for positive values
     * Most popular in modern deep learning
   - Sigmoid: f(z) = 1/(1 + e⁻ᶻ)
     * Output range: (0, 1), useful for probabilities
     * Issue: Vanishing gradients for large |z|
   - Tanh: f(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)
     * Output range: (-1, 1), zero-centered
   - Softmax: f(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
     * Used for multi-class classification
     * Outputs sum to 1 (probability distribution)

2. CONVOLUTIONAL NEURAL NETWORKS (CNNs)
   
   CNNs are specialized for processing grid-like data (images).
   
   Convolutional Layer:
   - Applies filters (kernels) to detect patterns
   - Each filter is a small matrix (e.g., 3×3, 5×5)
   - Convolution operation: (I * K)(i,j) = ΣₘΣₙ I(i+m, j+n) × K(m,n)
   
   Why Convolutions?
   - Parameter sharing: Same filter applied across entire image
   - Translation invariance: Detects features regardless of position
   - Hierarchical learning: Early layers detect edges, later layers detect complex patterns
   
   Pooling Layer:
   - Reduces spatial dimensions (downsampling)
   - Max pooling: Takes maximum value in each region
   - Average pooling: Takes average value
   - Purpose: Reduces computation, provides translation invariance
   
   Mathematical Example (Max Pooling 2×2):
   Input:  [[1, 2, 3, 4],     Output: [[6, 8],
            [5, 6, 7, 8],              [14, 16]]
            [9, 10, 11, 12],
            [13, 14, 15, 16]]

3. TRANSFER LEARNING
   
   Transfer learning leverages knowledge from pre-trained models.
   
   Concept:
   - Models trained on large datasets (e.g., ImageNet with 14M images) learn general features
   - Early layers learn universal patterns (edges, textures, shapes)
   - Later layers learn task-specific patterns
   
   Implementation Strategy:
   - Take pre-trained model (e.g., ResNet, VGG, EfficientNet)
   - Freeze early layers (keep learned weights)
   - Replace final classification layer for new task
   - Fine-tune: Optionally unfreeze and retrain later layers
   
   Why This Works:
   - Natural images share common low-level features
   - Medical images (MRI, CT) also have edges, textures
   - Fine-tuning adapts general features to specific domain

4. LOSS FUNCTIONS - Measuring Error
   
   Loss function quantifies how wrong the model's predictions are.
   
   Categorical Cross-Entropy (for multi-class classification):
   L = -Σᵢ yᵢ log(ŷᵢ)
   where:
   - yᵢ is true label (one-hot encoded)
   - ŷᵢ is predicted probability
   
   Intuition:
   - Penalizes confident wrong predictions heavily
   - log(ŷ) → -∞ as ŷ → 0 (wrong prediction)
   - log(ŷ) → 0 as ŷ → 1 (correct prediction)
   
   Binary Cross-Entropy (for binary classification):
   L = -[y log(ŷ) + (1-y) log(1-ŷ)]

5. OPTIMIZATION - Learning the Weights
   
   Gradient Descent:
   - Iteratively adjust weights to minimize loss
   - w_new = w_old - η × ∂L/∂w
   where η is learning rate
   
   Backpropagation:
   - Algorithm to compute gradients efficiently
   - Uses chain rule: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
   - Propagates error backwards through network
   
   Adam Optimizer (Adaptive Moment Estimation):
   - Combines momentum and adaptive learning rates
   - Maintains exponential moving averages of gradients
   - More stable and faster convergence than plain SGD
   - Formula:
     m_t = β₁m_{t-1} + (1-β₁)g_t  (momentum)
     v_t = β₂v_{t-1} + (1-β₂)g_t² (adaptive learning rate)
     w_t = w_{t-1} - η × m̂_t / (√v̂_t + ε)

6. REGULARIZATION - Preventing Overfitting
   
   Overfitting: Model memorizes training data but fails on new data
   
   Dropout:
   - Randomly deactivates neurons during training
   - Forces network to learn robust features
   - Rate 0.5 means 50% of neurons dropped each iteration
   
   Batch Normalization:
   - Normalizes layer inputs: x̂ = (x - μ_batch) / √(σ²_batch + ε)
   - Reduces internal covariate shift
   - Allows higher learning rates
   - Acts as regularizer
   
   L2 Regularization (Weight Decay):
   - Adds penalty for large weights: L_total = L_data + λΣw²
   - Encourages simpler models
   - Prevents overfitting

7. EVALUATION METRICS
   
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   - Simple but can be misleading with imbalanced data
   
   Precision = TP / (TP + FP)
   - Of all positive predictions, how many are correct?
   
   Recall = TP / (TP + FN)
   - Of all actual positives, how many did we find?
   
   F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   
   where:
   - TP = True Positives
   - TN = True Negatives
   - FP = False Positives
   - FN = False Negatives
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, VGG16, EfficientNetB0, MobileNetV2
)
from typing import Tuple, Optional
import numpy as np


class NeuropathologyModel:
    """
    Deep learning model for neuropathology detection from MRI images.
    
    Architecture Options:
    1. ResNet50: 
       - 50 layers with residual connections
       - Residual connection: y = F(x) + x (skip connections)
       - Solves vanishing gradient problem in very deep networks
       
    2. VGG16:
       - 16 layers, simple architecture
       - Sequential 3×3 convolutions
       - Good baseline, easier to understand
       
    3. EfficientNet:
       - Optimally scales width, depth, and resolution
       - Better accuracy with fewer parameters
       - State-of-the-art efficiency
       
    4. MobileNetV2:
       - Designed for mobile devices
       - Uses depthwise separable convolutions
       - Much faster, smaller model
    """
    
    def __init__(self, 
                 num_classes: int = 4,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 base_model: str = 'efficientnet',
                 trainable_layers: int = 20):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of pathology classes to predict
            input_shape: Shape of input images (height, width, channels)
            base_model: Pre-trained model to use ('resnet', 'vgg', 'efficientnet', 'mobilenet')
            trainable_layers: Number of layers to fine-tune (starting from the end)
        
        Theory - Model Initialization:
        - Weights initialized randomly affects convergence
        - Pre-trained weights provide better starting point
        - Xavier/He initialization used for new layers
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model_name = base_model
        self.trainable_layers = trainable_layers
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the complete model architecture.
        
        Returns:
            Compiled Keras model
            
        Architecture Design:
        1. Base Model (Pre-trained CNN): Extracts features from images
        2. Global Average Pooling: Reduces each feature map to single value
           - Reduces parameters dramatically
           - Makes model invariant to input size
           - Formula: GAP = (1/n) Σᵢ xᵢ for each feature map
        3. Dense Layers: Learns classification from extracted features
        4. Dropout: Prevents overfitting
        5. Output Layer: Produces class probabilities
        """
        # Load pre-trained base model
        base_model = self._get_base_model()
        
        # Freeze early layers (transfer learning strategy)
        # Theory: Early layers learn general features, keep them fixed
        base_model.trainable = False
        
        # Build complete model
        # Sequential API: Simple linear stack of layers
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Pre-trained base model (feature extractor)
            base_model,
            
            # Global Average Pooling
            # Reduces (7, 7, 2048) to (2048,) for ResNet
            # More robust than flatten, fewer parameters
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            
            # Batch Normalization
            # Normalizes activations, stabilizes training
            layers.BatchNormalization(name='bn1'),
            
            # First Dense Layer
            # Theory: Fully connected layer learns combinations of features
            # ReLU activation introduces non-linearity
            layers.Dense(512, activation='relu', name='fc1'),
            
            # Dropout for regularization
            # Randomly drops 50% of neurons during training
            # Prevents co-adaptation of features
            layers.Dropout(0.5, name='dropout1'),
            
            # Second Dense Layer
            # Additional capacity for complex patterns
            layers.Dense(256, activation='relu', name='fc2'),
            
            layers.BatchNormalization(name='bn2'),
            layers.Dropout(0.3, name='dropout2'),
            
            # Output Layer
            # Softmax produces probability distribution over classes
            # Output[i] represents P(class=i | input image)
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='neuropathology_model')
        
        self.model = model
        return model
    
    def _get_base_model(self) -> keras.Model:
        """
        Load the specified pre-trained base model.
        
        Returns:
            Pre-trained model without top classification layer
            
        Theory - ImageNet Pre-training:
        - Models trained on ImageNet (1000 classes, 1.2M images)
        - Learned features transfer well to medical imaging
        - include_top=False removes final classification layer
        - We add our own classification head for our specific task
        """
        if self.base_model_name == 'resnet':
            base_model = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'vgg':
            base_model = VGG16(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {self.base_model_name}")
        
        return base_model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> None:
        """
        Compile the model with loss function and optimizer.
        
        Args:
            learning_rate: Step size for weight updates
            optimizer: Optimization algorithm ('adam', 'sgd', 'rmsprop')
        
        Theory - Compilation:
        - Specifies how model should learn
        - Loss function: What to minimize
        - Optimizer: How to minimize
        - Metrics: What to monitor
        
        Learning Rate:
        - Too high: Oscillates, doesn't converge
        - Too low: Converges very slowly
        - 0.001 is good starting point for Adam
        - Can use learning rate schedules for better results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Choose optimizer
        if optimizer == 'adam':
            # Adam: Adaptive learning rate, momentum
            # Most popular choice for deep learning
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            # Stochastic Gradient Descent with momentum
            # More stable but slower convergence
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            # Root Mean Square Propagation
            # Good for recurrent networks
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            # Categorical cross-entropy for multi-class classification
            loss='categorical_crossentropy',
            # Track accuracy during training
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
    
    def fine_tune_model(self, learning_rate: float = 0.0001) -> None:
        """
        Unfreeze and fine-tune the base model.
        
        Args:
            learning_rate: Lower learning rate for fine-tuning
        
        Theory - Fine-Tuning:
        1. First train with frozen base model
           - Lets new layers learn appropriate features
           - Prevents destroying pre-trained weights
        
        2. Then unfreeze and fine-tune
           - Adapts pre-trained features to specific task
           - Uses lower learning rate to avoid drastic changes
           - Only fine-tune later layers (task-specific features)
        
        This two-stage approach gives best results:
        - Stage 1: Learn domain-specific classification
        - Stage 2: Adapt features for domain
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Unfreeze the base model
        base_model = self.model.layers[1]  # Assuming base model is second layer
        base_model.trainable = True
        
        # Freeze early layers, only train later layers
        # Theory: Early layers learn general features (edges, textures)
        #         Later layers learn task-specific features
        total_layers = len(base_model.layers)
        freeze_until = total_layers - self.trainable_layers
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        # Lower LR prevents destroying pre-trained weights
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet."
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Nothing to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of preprocessed images
        
        Returns:
            Array of class probabilities
            
        Theory - Inference:
        - Forward pass through network
        - No gradient computation (faster)
        - Dropout is disabled during inference
        - Batch normalization uses running statistics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        predictions = self.model.predict(images)
        return predictions
    
    def predict_class(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and confidence scores.
        
        Args:
            images: Batch of preprocessed images
        
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        probabilities = self.predict(images)
        predicted_classes = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predicted_classes, confidence_scores


def create_callbacks(model_path: str = 'models/best_model.h5') -> list:
    """
    Create training callbacks for monitoring and optimization.
    
    Returns:
        List of Keras callbacks
        
    Theory - Training Callbacks:
    
    1. ModelCheckpoint:
       - Saves best model during training
       - Monitors validation loss/accuracy
       - Keeps only best version
    
    2. EarlyStopping:
       - Stops training when validation loss stops improving
       - Prevents overfitting
       - Patience: Number of epochs to wait before stopping
    
    3. ReduceLROnPlateau:
       - Reduces learning rate when plateau detected
       - Helps escape local minima
       - Learning rate annealing improves final accuracy
    
    4. TensorBoard:
       - Visualizes training progress
       - Plots loss, accuracy curves
       - Examines model architecture
    """
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks
