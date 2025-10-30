# Deep Learning Tutorial: Complete Learning Path

## Overview

This is a comprehensive, textbook-level tutorial on deep learning, grounded in the brain tumor MRI classification project (neuropathology-dl). The tutorial covers theory from first principles through state-of-the-art practice, with rigorous mathematical derivations, working code examples, and detailed mappings to the project codebase.

## Tutorial Structure

Each session follows a 6-part structure:
- **Part A**: Intuition and Geometric Approach
- **Part B**: Formal Theory and Definitions
- **Part C**: Mathematical Derivations (complete, step-by-step)
- **Part D**: Code Implementation and Demonstrations
- **Part E**: Project Mapping (to neuropathology codebase with line numbers)
- **Part F**: Advanced Topics (marked with asterisk * for optional deep dives)

## Learning Objectives

By the end of this tutorial, you will:

1. **Master Deep Learning Foundations**: Understand neural networks, backpropagation, optimization, and regularization from first principles
2. **Implement from Scratch**: Build neural networks, CNNs, and training algorithms without high-level frameworks
3. **Use Modern Frameworks**: Understand TensorFlow/Keras internals and use them effectively
4. **Apply Transfer Learning**: Leverage pretrained models and fine-tune for specific tasks
5. **Handle Real-World Data**: Deal with class imbalance, data augmentation, and medical imaging specifics
6. **Evaluate Rigorously**: Use appropriate metrics, interpret results, and avoid common pitfalls
7. **Deploy Responsibly**: Understand ethical considerations, especially for medical AI
8. **Read Research Papers**: Have the mathematical and conceptual foundation to understand current literature

## Sessions

### Session 01: Neural Network Fundamentals ✓ (In Progress)
**Status**: Parts A, B, C complete

**Topics**:
- The learning problem in high dimensions
- Biological inspiration and mathematical abstraction
- Perceptrons and linear classifiers
- Multilayer networks and universal approximation
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Forward propagation
- Loss functions (cross-entropy, MSE)
- Backpropagation derivation from first principles
- Gradient computation and chain rule
- Vanishing/exploding gradients
- Matrix calculus and vectorization

**Project Connection**: Foundation for understanding `/src/models/neuropathology_model.py` layers

**Files**:
- `session01_neural_networks/part_a_geometric_intuition.md` ✓
- `session01_neural_networks/part_b_formal_theory.md` ✓
- `session01_neural_networks/part_c_mathematical_derivations.md` ✓
- `session01_neural_networks/part_d_code_implementation.md` (TODO)
- `session01_neural_networks/part_e_project_mapping.md` (TODO)
- `session01_neural_networks/part_f_advanced_topics.md` (TODO)

### Session 02: Convolutional Neural Networks
**Topics**:
- Convolution operation (mathematical definition)
- Discrete convolution vs. cross-correlation
- Padding, stride, dilation
- Receptive fields (derivation and visualization)
- Pooling operations
- Hierarchical feature learning
- Parameter sharing and translation invariance
- CNN architectures (LeNet, AlexNet, VGG)
- Backpropagation through convolutions
- Implementation from scratch
- Visualization of learned filters

**Project Connection**: Understanding MobileNetV2 base in our classifier

**Files** (TODO):
- Parts A-F following same structure

### Session 03: Transfer Learning and MobileNetV2
**Topics**:
- Feature extraction vs. fine-tuning
- Domain shift and adaptation
- ImageNet pretraining
- When transfer learning works
- MobileNetV2 architecture deep dive:
  - Depthwise separable convolutions (math and efficiency)
  - Inverted residuals
  - Linear bottlenecks
  - Architecture scaling
- Efficient CNNs (comparison with ResNet, EfficientNet)
- Two-stage training strategy
- Learning rate selection for fine-tuning
- Layer freezing strategies

**Project Connection**: Core of `/src/models/neuropathology_model.py` base model

**Files** (TODO):
- Parts A-F

### Session 04: Training Mechanics and Optimization
**Topics**:
- Gradient descent variants (batch, mini-batch, stochastic)
- Momentum and Nesterov accelerated gradient
- Adaptive learning rates: AdaGrad, RMSprop, Adam
- Adam derivation and intuition
- Learning rate schedules (step decay, cosine annealing, warm restarts)
- Initialization strategies (Xavier, He)
- Batch size effects on generalization
- Gradient clipping
- Second-order methods (Newton, L-BFGS) *
- Natural gradient descent *
- Optimization landscape analysis *

**Project Connection**: Optimizers and learning rates in `train.py`

**Files** (TODO):
- Parts A-F

### Session 05: Regularization Techniques
**Topics**:
- Overfitting: causes and detection
- L1 and L2 regularization (derivation, geometric interpretation)
- Dropout: intuition, mathematics, and implementation
- Batch normalization:
  - Internal covariate shift
  - Normalization math (train vs. inference)
  - Backpropagation through batch norm
  - Layer normalization, group normalization *
- Data augmentation for images:
  - Geometric transforms (rotation, flip, zoom)
  - Color jittering
  - Cutout and Mixup *
  - Medical imaging considerations
- Early stopping
- Ensemble methods *
- Test-time augmentation *

**Project Connection**: Regularization in model head, data augmentation in `data_loader.py`

**Files** (TODO):
- Parts A-F

### Session 06: Evaluation and Class Imbalance
**Topics**:
- Classification metrics:
  - Accuracy and its limitations
  - Precision, recall, F1-score (derivation)
  - ROC curves and AUC
  - Precision-recall curves
  - Confusion matrices
- Multi-class metrics (micro, macro, weighted averaging)
- Class imbalance:
  - Why it matters
  - Class weighting (derivation)
  - Oversampling and undersampling
  - SMOTE and variants *
- Calibration:
  - Expected calibration error
  - Temperature scaling
- Uncertainty quantification *
- Statistical significance testing *

**Project Connection**: Evaluation in `train.py` and `evaluate_model.py`

**Files** (TODO):
- Parts A-F

### Session 07: Practical Training Craft
**Topics**:
- Dataset splitting (train/val/test)
- Cross-validation strategies
- Hyperparameter tuning:
  - Grid search, random search
  - Bayesian optimization *
- Callbacks:
  - EarlyStopping (implementation)
  - ModelCheckpoint
  - ReduceLROnPlateau
  - Custom callbacks
- Debugging training:
  - Sanity checks
  - Overfitting a single batch
  - Gradient monitoring
  - Activation visualization
- Reproducibility:
  - Random seeds
  - Deterministic operations
- Experiment tracking and logging
- Model versioning
- Computational considerations (CPU vs. GPU)

**Project Connection**: Complete walkthrough of `train.py`

**Files** (TODO):
- Parts A-F

### Session 08: Interpretability and Explainability
**Topics**:
- Why interpretability matters (especially for medical AI)
- Activation maximization
- Saliency maps (vanilla and smoothgrad)
- Gradient-weighted Class Activation Mapping (Grad-CAM):
  - Mathematical derivation
  - Implementation
  - Application to brain tumor classification
- Integrated gradients *
- LIME and SHAP *
- Attention mechanisms and visualization *
- Layer-wise relevance propagation *
- Medical imaging caveats

**Project Connection**: Adding interpretability to the neuropathology project

**Files** (TODO):
- Parts A-F

### Session 09: Robustness and Distribution Shift
**Topics**:
- Train/test distribution mismatch
- Domain adaptation techniques *
- Adversarial examples:
  - FGSM, PGD attacks
  - Adversarial training
- Robustness metrics
- Data leakage prevention
- Covariate shift, concept drift
- Out-of-distribution detection *

**Project Connection**: Making the classifier more robust

**Files** (TODO):
- Parts A-F

### Session 10: Ethics and Responsible AI in Medicine
**Topics**:
- Fairness and bias in medical AI
- Dataset biases and their impact
- Generalization risks
- Clinical validation requirements
- Regulatory considerations (FDA, CE marking)
- Privacy (HIPAA, GDPR)
- Federated learning for privacy *
- Model transparency and documentation
- Responsible deployment

**Project Connection**: Ethical considerations for the neuropathology classifier

**Files** (TODO):
- Parts A-F

## Appendices

### Appendix A: Mathematical Prerequisites
- Linear algebra review (vectors, matrices, eigenvalues)
- Calculus review (derivatives, gradients, chain rule)
- Probability theory (distributions, expectation, Bayes' theorem)
- Information theory (entropy, KL divergence, mutual information)
- Optimization theory (convexity, Lagrange multipliers)

### Appendix B: Python and NumPy Essentials
- NumPy broadcasting
- Efficient vectorization
- Memory management
- Profiling code

### Appendix C: TensorFlow/Keras Deep Dive
- Computational graphs
- Automatic differentiation
- Custom layers and loss functions
- tf.data API for efficient data loading
- Distributed training *
- Mixed precision training *

### Appendix D: Advanced Architectures *
- Residual networks (ResNet) and skip connections
- DenseNet and dense connections
- EfficientNet and neural architecture search
- Vision transformers (ViT) *
- U-Net for medical image segmentation *

### Appendix E: Beyond Classification *
- Object detection (R-CNN family, YOLO, RetinaNet)
- Semantic segmentation
- Instance segmentation
- Multi-task learning
- Few-shot learning
- Self-supervised learning
- Contrastive learning (SimCLR, MoCo)

### Appendix F: Research Frontiers *
- Neural architecture search
- Meta-learning
- Continual learning
- Causal inference in deep learning
- Neurosymbolic AI
- Geometric deep learning

## Notation Conventions

Throughout this tutorial, we use consistent mathematical notation:

**Scalars**: Lowercase italic letters ($x$, $y$, $z$, $\alpha$, $\beta$)

**Vectors**: Lowercase bold letters ($\mathbf{x}$, $\mathbf{w}$, $\mathbf{b}$)
- Column vectors by default
- $\mathbf{x}^T$ denotes transpose (row vector)

**Matrices**: Uppercase bold letters ($\mathbf{W}$, $\mathbf{X}$, $\mathbf{A}$)

**Tensors**: Uppercase bold calligraphic letters ($\mathcal{X}$, $\mathcal{W}$) or explicit notation with indices

**Sets**: Uppercase italic letters ($X$, $Y$) or calligraphic ($\mathcal{D}$, $\mathcal{H}$)

**Functions**: Lowercase or uppercase italic ($f$, $g$, $F$, $L$)

**Gradients**: $\nabla f$ or $\frac{\partial f}{\partial x}$

**Special notation**:
- $\|\mathbf{x}\|$: Euclidean (L2) norm
- $\|\mathbf{x}\|_1$: L1 norm
- $\mathbf{x} \odot \mathbf{y}$: Element-wise (Hadamard) product
- $\mathbb{E}[\cdot]$: Expectation
- $\mathbb{P}(\cdot)$: Probability
- $\sim$: "distributed as" (e.g., $x \sim \mathcal{N}(0, 1)$)
- $:=$: Definition
- $\propto$: Proportional to
- $\mathbb{1}[\cdot]$: Indicator function (1 if condition true, 0 otherwise)
- $\arg\min$, $\arg\max$: Argument that minimizes/maximizes

**Layer notation** (for neural networks):
- Superscript $(\ell)$: Layer index (e.g., $\mathbf{W}^{(2)}$ is weights of layer 2)
- Subscript $i$: Neuron/element index (e.g., $a_i^{(2)}$ is activation of neuron $i$ in layer 2)
- Subscript $ij$: Matrix index (e.g., $W_{ij}$ is weight from neuron $j$ to neuron $i$)
- $\mathbf{a}^{(\ell)}$: Activation (output) of layer $\ell$
- $\mathbf{z}^{(\ell)}$: Pre-activation (logit) of layer $\ell$
- $\boldsymbol{\delta}^{(\ell)} = \frac{\partial L}{\partial \mathbf{z}^{(\ell)}}$: Error signal

## Prerequisites

**Required**:
- Undergraduate linear algebra (vectors, matrices, eigenvalues)
- Calculus (derivatives, partial derivatives, chain rule)
- Basic probability (random variables, expectation, common distributions)
- Python programming
- Comfort with mathematical notation and proofs

**Helpful but not required**:
- Prior machine learning experience
- Optimization theory
- Information theory
- Deep learning frameworks (TensorFlow, PyTorch)

All necessary mathematical background is reviewed on-the-fly in the tutorial.

## How to Use This Tutorial

### For a Comprehensive Course (Full Day or Week-Long Study):

1. **Day 1**: Sessions 01-02 (Neural networks fundamentals, CNNs)
2. **Day 2**: Sessions 03-04 (Transfer learning, optimization)
3. **Day 3**: Sessions 05-06 (Regularization, evaluation)
4. **Day 4**: Sessions 07-08 (Practical training, interpretability)
5. **Day 5**: Sessions 09-10 (Robustness, ethics)

Work through each part (A-F) sequentially for each session.

### For Targeted Learning:

- **Want to understand the math**: Focus on Parts B and C
- **Want to implement from scratch**: Focus on Parts D and E
- **Want to apply to your project**: Focus on Parts E and F
- **Want advanced topics**: Read Part F of each session and starred appendices

### For Project-Based Learning:

Start with Session 01 Part E to understand the project structure, then work backward through the theory as needed.

### Code Execution:

All code examples in Part D are designed to run in Google Colab with CPU-only environments. Simply copy-paste into Colab cells. Visualization outputs are saved as images and referenced in the markdown.

## Project Repository Structure

The neuropathology-dl project this tutorial is based on:

```
neuropathology-dl/
├── train.py                 # Main training script
├── evaluate_model.py        # Evaluation utilities
├── app.py                   # Streamlit web interface
├── requirements.txt         # Dependencies
├── src/
│   ├── models/
│   │   └── neuropathology_model.py   # Model architecture
│   ├── data/
│   │   └── data_loader.py            # Data loading and augmentation
│   └── utils/
│       └── helpers.py                # Utility functions
└── learn/                   # This tutorial
    ├── README.md           # This file
    ├── session01_neural_networks/
    ├── session02_cnns/
    └── ...
```

## Status

**Completed**:
- Tutorial framework and structure
- Session 01 Parts A, B, C (Neural Network Fundamentals)

**In Progress**:
- Session 01 Parts D, E, F

**TODO**:
- Sessions 02-10 (all parts)
- All appendices
- Visualization scripts
- Complete project code mapping

## Contributing

This tutorial is a living document. Contributions, corrections, and suggestions are welcome. Each markdown file is self-contained with complete derivations and can be read independently or as part of the sequence.

## License

This tutorial content is released under MIT License, consistent with the neuropathology-dl project.

## Acknowledgments

This tutorial is grounded in the practical neuropathology brain tumor MRI classifier project, providing real-world context for every theoretical concept.

---

**Next Steps**: Continue with Session 01 Part D (Code Implementation) to implement neural networks from scratch in NumPy, then Part E to map concepts to the actual project codebase.
