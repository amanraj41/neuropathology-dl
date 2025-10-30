# Session 01 Part F: Neural Networks - Advanced Topics*

## Table of Contents

1. [The Optimization Landscape*](#the-optimization-landscape)
2. [Loss Surface Visualization*](#loss-surface-visualization)
3. [Representation Learning and Feature Geometry*](#representation-learning-and-feature-geometry)
4. [The Lottery Ticket Hypothesis*](#the-lottery-ticket-hypothesis)
5. [Double Descent and Overparameterization*](#double-descent-and-overparameterization)
6. [Neural Tangent Kernels*](#neural-tangent-kernels)
7. [Implicit Regularization of Gradient Descent*](#implicit-regularization-of-gradient-descent)
8. [Alternative Training Paradigms*](#alternative-training-paradigms)
9. [Connections to Kernel Methods*](#connections-to-kernel-methods)
10. [Future Research Directions*](#future-research-directions)

---

*Note: Topics marked with asterisk are advanced material that can be skipped on first reading. They provide deeper theoretical understanding and connections to research frontiers.*

## The Optimization Landscape*

Neural network training is a high-dimensional nonconvex optimization problem. Understanding the loss landscape—the geometry of the loss function as a function of parameters—provides insight into why gradient descent succeeds despite theoretical challenges.

### Local Minima, Saddle Points, and Plateaus

For a function $L: \mathbb{R}^d \to \mathbb{R}$ (our loss), critical points satisfy $\nabla L(\boldsymbol{\theta}) = \mathbf{0}$.

**Classification of critical points** using the Hessian $\mathbf{H} = \nabla^2 L$:

1. **Local minimum**: All eigenvalues of $\mathbf{H}$ are positive
   - Loss increases in all directions
   - Gradient descent can get stuck here

2. **Local maximum**: All eigenvalues of $\mathbf{H}$ are negative
   - Loss decreases in all directions
   - Rarely encountered during training

3. **Saddle point**: $\mathbf{H}$ has both positive and negative eigenvalues
   - Loss decreases in some directions, increases in others
   - $\nabla L = 0$ but not a minimum

**Key insight**: In high dimensions, saddle points vastly outnumber local minima.

**Why?** Suppose each eigenvalue of the Hessian is independently positive or negative with probability 0.5. The probability that all $d$ eigenvalues are positive (local minimum) is $2^{-d}$. For $d = 10^6$, this is astronomically small.

**Empirical finding**: For overparameterized neural networks, most critical points are saddle points, and gradient-based methods can escape them.

### Mode Connectivity

**Discovery**: Different local minima found by gradient descent can often be connected by a path through weight space along which the loss remains low.

**Formalization**: Given two sets of parameters $\boldsymbol{\theta}_A$ and $\boldsymbol{\theta}_B$ (both achieving low loss), there often exists a path $\boldsymbol{\theta}(t)$ for $t \in [0, 1]$ such that:
- $\boldsymbol{\theta}(0) = \boldsymbol{\theta}_A$
- $\boldsymbol{\theta}(1) = \boldsymbol{\theta}_B$
- $L(\boldsymbol{\theta}(t))$ remains low for all $t$

**Implication**: The loss landscape has flat valleys rather than isolated basins. This may explain why different random initializations lead to similar generalization performance.

### Sharpness and Generalization

**Observation**: Minima found by large-batch SGD tend to be "sharp" (high curvature), while small-batch SGD finds "flat" minima (low curvature).

**Measure of sharpness**: 
$$\text{Sharpness} = \max_{\|\boldsymbol{\epsilon}\| \leq \rho} L(\boldsymbol{\theta}^* + \boldsymbol{\epsilon}) - L(\boldsymbol{\theta}^*)$$

where $\boldsymbol{\theta}^*$ is a local minimum and $\rho$ is a small radius.

**Connection to generalization**: Flat minima generalize better. Intuitively, if the loss is flat around $\boldsymbol{\theta}^*$, small perturbations (e.g., from test data) don't increase loss much.

**Why small batches prefer flat minima**: Noise in small-batch gradients acts as implicit regularization, preventing convergence to sharp minima.

## Loss Surface Visualization*

Visualizing high-dimensional loss surfaces is challenging, but techniques exist to gain intuition.

### Random Direction Visualization

Project the loss surface onto a random 1D line:

$$L(\alpha) = L(\boldsymbol{\theta}^* + \alpha \mathbf{d})$$

where $\mathbf{d}$ is a random unit vector.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_1d(model, data, theta_star, direction, alpha_range=(-1, 1), num_points=50):
    """
    Visualize loss along a random direction from current parameters.
    
    Args:
        model: Neural network model
        data: (X, y) tuple of data
        theta_star: Current parameters (flattened)
        direction: Random direction vector (unit norm)
        alpha_range: Range of alpha values to plot
        num_points: Number of points to sample
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = []
    
    X, y = data
    
    for alpha in alphas:
        # Perturb parameters: theta = theta_star + alpha * direction
        theta_perturbed = theta_star + alpha * direction
        
        # Set model parameters
        model.set_weights_flat(theta_perturbed)
        
        # Compute loss
        loss = model.compute_loss(X, y)
        losses.append(loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, losses, linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', label='Current parameters')
    plt.xlabel('α (step along random direction)')
    plt.ylabel('Loss')
    plt.title('Loss Surface along Random Direction')
    plt.legend()
    plt.grid(True)
    plt.savefig('../visualizations/loss_1d_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 2D Loss Contours

Project onto a random 2D plane:

$$L(\alpha, \beta) = L(\boldsymbol{\theta}^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$$

where $\mathbf{d}_1, \mathbf{d}_2$ are orthogonal unit vectors.

```python
def visualize_loss_2d(model, data, theta_star, direction1, direction2, 
                      range_alpha=(-1, 1), range_beta=(-1, 1), num_points=20):
    """
    Visualize loss surface as 2D contour plot.
    """
    alphas = np.linspace(range_alpha[0], range_alpha[1], num_points)
    betas = np.linspace(range_beta[0], range_beta[1], num_points)
    
    loss_surface = np.zeros((num_points, num_points))
    
    X, y = data
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta_perturbed = theta_star + alpha * direction1 + beta * direction2
            model.set_weights_flat(theta_perturbed)
            loss_surface[i, j] = model.compute_loss(X, y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(alphas, betas, loss_surface, levels=20)
    plt.colorbar(label='Loss')
    plt.scatter([0], [0], c='red', s=100, marker='*', label='Current parameters')
    plt.xlabel('α (first direction)')
    plt.ylabel('β (second direction)')
    plt.title('2D Loss Surface Contours')
    plt.legend()
    plt.savefig('../visualizations/loss_2d_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
```

**Observations from visualizations**:
- Early in training: Loss surface is rough with many local minima
- Late in training: Smoother surface, convex near the minimum
- Overparameterized networks: Very flat regions (many equivalent solutions)

## Representation Learning and Feature Geometry*

Neural networks don't just classify; they learn representations. Understanding the geometry of these learned representations provides insight into how networks make decisions.

### Feature Space Geometry

For a trained network, consider the activations at a hidden layer $\ell$: $\mathbf{a}^{(\ell)}(\mathbf{x}) \in \mathbb{R}^{n_\ell}$.

This maps each input $\mathbf{x}$ to a point in $\mathbb{R}^{n_\ell}$. We call this the **feature space** or **representation space** at layer $\ell$.

**Key questions**:
1. Are examples from the same class clustered together?
2. Are different classes well-separated?
3. Is the representation linearly separable?

### Linear Separability of Learned Representations

**Definition**: A dataset is **linearly separable** in feature space if there exists a hyperplane that perfectly separates the classes.

For our brain tumor classifier, after the dense layers, we have a 256-dimensional feature space (output of the second dense layer). Ideally, the 17 tumor types should form 17 well-separated clusters in this space.

**Measuring separability**: Train a linear classifier (logistic regression) on the learned features. If it achieves high accuracy, the representation is linearly separable.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def measure_linear_separability(model, data, layer_name='fc2'):
    """
    Measure how linearly separable the learned representations are.
    
    Args:
        model: Trained neural network
        data: (X_train, y_train, X_test, y_test) tuple
        layer_name: Name of layer to extract features from
        
    Returns:
        Linear probe accuracy (accuracy of linear classifier on features)
    """
    X_train, y_train, X_test, y_test = data
    
    # Extract features from specified layer
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    train_features = feature_extractor.predict(X_train)
    test_features = feature_extractor.predict(X_test)
    
    # Train linear classifier
    linear_probe = LogisticRegression(max_iter=1000, multi_class='multinomial')
    linear_probe.fit(train_features, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, linear_probe.predict(train_features))
    test_acc = accuracy_score(y_test, linear_probe.predict(test_features))
    
    print(f"Linear probe on {layer_name} features:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    
    return test_acc
```

**Typical findings**:
- Early layers: Not linearly separable (features are low-level, mixed)
- Middle layers: Increasingly separable
- Final hidden layer: Highly linearly separable (ideal for classification)

This validates the hierarchical feature learning hypothesis.

### Dimensionality of Representation Manifolds

**Hypothesis**: Natural data (like images) lies on a low-dimensional manifold embedded in high-dimensional space.

For example, brain MRI images are $224 \times 224 \times 3 = 150{,}528$ dimensional, but the space of "natural brain MRI images" has much lower intrinsic dimensionality—perhaps a few hundred dimensions.

**Measuring intrinsic dimensionality** using PCA:

```python
from sklearn.decomposition import PCA

def estimate_intrinsic_dimensionality(features, variance_threshold=0.95):
    """
    Estimate intrinsic dimensionality using PCA.
    
    Returns the number of principal components needed to explain
    variance_threshold of the total variance.
    """
    pca = PCA()
    pca.fit(features)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Intrinsic dimensionality (for {variance_threshold*100}% variance): {intrinsic_dim}")
    print(f"Original dimensionality: {features.shape[1]}")
    
    return intrinsic_dim
```

**Insight**: Good representations should have low intrinsic dimensionality for each class, with classes separated in a higher-dimensional embedding space.

## The Lottery Ticket Hypothesis*

A fascinating recent discovery about network pruning and initialization.

### Statement of the Hypothesis

**Lottery Ticket Hypothesis** (Frankle & Carbin, 2019): A randomly-initialized, dense neural network contains a subnetwork (a "winning ticket") that, when trained in isolation, can match the accuracy of the original network.

Moreover, this winning ticket performs well only when initialized with the same initial weights as the original network.

### Implications

1. **Not all parameters are necessary**: A small fraction of weights suffices for good performance

2. **Initialization matters**: The same subnetwork with different initialization performs poorly

3. **Training discovers, not creates**: Training doesn't create the winning ticket; it discovers which weights are important

### Finding Winning Tickets

**Iterative magnitude pruning** (IMP):

1. Randomly initialize network: $\boldsymbol{\theta}_0$
2. Train for $T$ iterations: $\boldsymbol{\theta}_0 \to \boldsymbol{\theta}_T$
3. Prune $p$% of weights with smallest magnitude
4. Reset remaining weights to $\boldsymbol{\theta}_0$
5. Repeat from step 2

After several rounds, you're left with a sparse subnetwork that trains faster and generalizes better.

**Relevance to our project**: For deployment on resource-constrained devices (e.g., medical imaging on tablets), we could use lottery ticket pruning to create a smaller, faster model with similar accuracy.

### Open Questions

- Why do winning tickets exist?
- Can we find them without training the full network first?
- Do winning tickets transfer across datasets?

## Double Descent and Overparameterization*

Classical statistical learning theory predicts: as model complexity increases, training error decreases but test error eventually increases (overfitting). This gives a U-shaped test error curve.

### The Double Descent Phenomenon

**Observation**: For modern neural networks, test error exhibits **double descent**:

1. **Classical regime** (underparameterized): Test error decreases then increases
2. **Interpolation threshold**: Peak in test error when #parameters ≈ #training examples
3. **Modern regime** (overparameterized): Test error decreases again!

```
Test Error
    │     Classical      │    Modern
    │       regime       │   regime
    │                   ╱│╲
    │                 ╱  │  ╲
    │               ╱    │    ╲___
    │             ╱      │
    │___________╱        │
    │                    │
    └────────────────────┼───────────→ Model Complexity
                Interpolation         (#parameters)
                 threshold
```

### Why Does This Happen?

**Intuition**: In the overparameterized regime, there are many solutions that fit the training data perfectly (interpolate). Gradient descent implicitly selects among these solutions, preferring those with good inductive bias.

**Min-norm interpolation**: Among all solutions that perfectly fit the training data, gradient descent finds the one with minimum $\ell^2$ norm. This minimum-norm solution often generalizes well.

**Mathematical connection**: In the infinite-width limit, neural networks behave like linear models in a high-dimensional feature space, and the implicit regularization of gradient descent becomes explicit.

### Implications for Practice

1. **More parameters can help**: Don't fear overparameterization
2. **Implicit regularization**: Gradient descent has inductive biases that promote generalization
3. **Conventional wisdom fails**: Classical complexity measures (VC dimension, Rademacher complexity) don't explain generalization in deep learning

## Neural Tangent Kernels*

A theoretical framework connecting neural networks to kernel methods in the infinite-width limit.

### The Neural Tangent Kernel (NTK)

For a network $f(\mathbf{x}; \boldsymbol{\theta})$, define the **neural tangent kernel**:

$$K(\mathbf{x}, \mathbf{x}') = \left\langle \frac{\partial f(\mathbf{x}; \boldsymbol{\theta}_0)}{\partial \boldsymbol{\theta}}, \frac{\partial f(\mathbf{x}'; \boldsymbol{\theta}_0)}{\partial \boldsymbol{\theta}} \right\rangle$$

This is the inner product of the gradients at initialization.

**Remarkable property**: For infinitely wide networks, the NTK remains constant during training!

### Training Dynamics in the NTK Regime

For infinite width, the network training dynamics become:

$$\frac{df(\mathbf{x}; \boldsymbol{\theta}_t)}{dt} = -\eta \sum_{i=1}^{N} K(\mathbf{x}, \mathbf{x}_i) (f(\mathbf{x}_i; \boldsymbol{\theta}_t) - y_i)$$

This is a **linear differential equation** in function space, analogous to kernel ridge regression.

**Implication**: In the infinite-width limit, neural network training is equivalent to kernel regression with a specific kernel (the NTK).

### Connections to Practice

**Finite-width networks**: Behave differently from NTK regime
- Features evolve during training
- Nonlinear dynamics
- More expressiveness

**Practical takeaway**: While NTK provides theoretical insight, finite-width networks (which we use in practice) exhibit richer learning dynamics not captured by the NTK theory.

---

*Note: This completes Part F of Session 01. These advanced topics provide connections to current research and deeper theoretical understanding. They can be revisited after completing the main tutorial sessions.*
