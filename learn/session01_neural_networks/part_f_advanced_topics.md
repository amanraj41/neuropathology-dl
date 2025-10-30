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

### Empirical Loss Landscape Visualization*

While theoretical analysis is valuable, empirical visualization provides intuition.

**Method**: Fix two random directions $\mathbf{d}_1, \mathbf{d}_2$ in parameter space. Plot loss as a function of $\alpha, \beta$:
$$L(\alpha, \beta) = L(\theta^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$$

where $\theta^*$ is a trained model's parameters.

**Implementation sketch**:
```python
def visualize_loss_landscape(model, data, alpha_range, beta_range):
    """Plot 2D slice of loss surface."""
    theta_star = get_model_params(model)
    d1 = random_direction(theta_star.shape)
    d2 = random_direction(theta_star.shape)
    
    losses = np.zeros((len(alpha_range), len(beta_range)))
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            theta = theta_star + alpha * d1 + beta * d2
            set_model_params(model, theta)
            losses[i, j] = evaluate_loss(model, data)
    
    # Plot contour
    plt.contour(alpha_range, beta_range, losses.T, levels=20)
    plt.xlabel('α (direction 1)')
    plt.ylabel('β (direction 2)')
    plt.title('Loss Landscape')
```

**Observations from research** (Goodfellow et al., 2015; Li et al., 2018):
- Loss surfaces are highly non-convex but not "pathologically bad"
- Wide, flat minima generalize better than sharp minima
- Successful training often finds "mode-connected" minima

## Representation Learning and Feature Geometry*

Neural networks learn hierarchical representations. Understanding this geometry provides deep insights.

### The Manifold Hypothesis

**Hypothesis**: Natural data (images, text, audio) lies on low-dimensional manifolds embedded in high-dimensional ambient space.

For brain MRIs:
- **Ambient space**: $\mathbb{R}^{224 \times 224 \times 3} = \mathbb{R}^{150528}$
- **Data manifold**: Much lower dimensional, perhaps $\mathbb{R}^{100}$ or $\mathbb{R}^{200}$

**Why**: Brain anatomy is constrained—not all 150,528-dimensional vectors correspond to realistic brain scans. The manifold captures anatomically valid variations.

**Neural networks as manifold learners**: Each layer maps input to a new representation space, progressively linearizing the manifold to make classes separable.

### Feature Geometry: From Raw Pixels to Abstract Concepts

**Visualization technique**: t-SNE or UMAP projections of intermediate layer activations.

**Expected progression** (for our classifier):
1. **Early layers** (MobileNetV2 conv blocks 1-5): Cluster by texture, brightness, contrast. Different scan types (T1, T1C+, T2) might cluster separately.

2. **Middle layers** (conv blocks 6-12): Cluster by anatomical region. Scans showing similar structures group together regardless of pathology.

3. **Late layers** (conv blocks 13-17): Cluster by pathology. Gliomas separate from meningiomas, schwannomas form distinct clusters.

4. **Dense layers** (fc1, fc2): Classes become linearly separable. Decision boundaries are hyperplanes in this 256-dimensional space.

**Mathematical formalization**:

Define the **representation** at layer $\ell$ as $\mathbf{h}^{(\ell)} = f^{(\ell)}(\mathbf{x}; \theta)$.

**Intra-class variance**:
$$\text{Var}_{\text{intra}} = \mathbb{E}_{y} \left[ \mathbb{E}_{\mathbf{x} | y}[\|\mathbf{h}^{(\ell)} - \mu_y\|^2] \right]$$

where $\mu_y = \mathbb{E}_{\mathbf{x} | y}[\mathbf{h}^{(\ell)}]$ is the class centroid.

**Inter-class variance**:
$$\text{Var}_{\text{inter}} = \mathbb{E}_{y}[\|\mu_y - \mu_{\text{global}}\|^2]$$

where $\mu_{\text{global}} = \mathbb{E}_{\mathbf{x}}[\mathbf{h}^{(\ell)}]$.

**Good representations**: High $\text{Var}_{\text{inter}}$ (classes far apart) and low $\text{Var}_{\text{intra}}$ (tight within-class clusters).

### Linear Separability and SVM Probing

**Question**: At which layer do representations become linearly separable?

**Method**: Train a linear SVM (Support Vector Machine) on frozen representations from layer $\ell$:
$$\text{SVM}: \mathbf{h}^{(\ell)} \mapsto y$$

If SVM achieves high accuracy, representations are linearly separable.

**Expected results**:
- Early layers: Low SVM accuracy (~20% for 17 classes, near random)
- Middle layers: Moderate accuracy (~50-60%)
- Final dense layer: High accuracy (~90%+, approaching full model performance)

**Interpretation**: The network progressively linearizes the problem. The final softmax layer merely applies a linear classifier to well-separated features.

## The Lottery Ticket Hypothesis*

**Paper**: Frankle & Carbin (2019), "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Subnetworks"

**Hypothesis**: Dense neural networks contain sparse subnetworks (winning tickets) that, when trained in isolation from the start, achieve comparable accuracy to the full network.

### Formal Statement

For a dense network with initial weights $\theta_0$ and final trained weights $\theta_f$:

There exists a binary mask $\mathbf{m} \in \{0,1\}^{|\theta|}$ with sparsity $s$ (e.g., 20% of weights kept) such that:

$$\text{Accuracy}(\mathbf{m} \odot \theta_0 \text{ trained to convergence}) \approx \text{Accuracy}(\theta_f)$$

where $\odot$ is element-wise multiplication (masking).

**Key point**: The subnetwork must use the *original initialization* $\theta_0$, not a random initialization.

### Finding Winning Tickets: Iterative Magnitude Pruning

**Algorithm**:
1. Randomly initialize network: $\theta_0$
2. Train to convergence: $\theta_0 \to \theta_f$
3. Prune $p\%$ of weights with smallest magnitude: Create mask $\mathbf{m}$
4. Reset remaining weights to original values: $\tilde{\theta}_0 = \mathbf{m} \odot \theta_0$
5. Train pruned network: $\tilde{\theta}_0 \to \tilde{\theta}_f$
6. Evaluate accuracy
7. If target sparsity not reached, repeat from step 3

**Results from original paper**:
- Found subnetworks at 10-20% sparsity matching full network accuracy
- Subnetworks train faster (fewer parameters to update)
- Crucial: Must rewind to original initialization, not random re-initialization

### Implications

**Theoretical**: Suggests that:
- Overparameterization helps optimization (more "lottery tickets" to find)
- Initialization contains critical information for trainability
- Many parameters are redundant; networks could be more efficient

**Practical**: Enables network compression without accuracy loss.

**Connection to our project**: Our dense layers have 655K and 131K parameters. Lottery ticket hypothesis suggests we could potentially prune 80% of these without accuracy loss, significantly reducing model size for deployment.

## Double Descent and Overparameterization*

**Classical statistical learning theory**: More parameters → better training fit but worse generalization (overfitting). Test error should monotonically increase beyond optimal model complexity.

**Empirical observation** (Belkin et al., 2019): For modern deep learning, test error exhibits **double descent**:
1. **Classical regime**: Increasing complexity initially improves test error
2. **Interpolation threshold**: At capacity = training samples, test error peaks
3. **Modern regime**: Further increasing complexity *decreases* test error!

### The Double Descent Curve

**X-axis**: Model complexity (number of parameters or training epochs)

**Y-axis**: Test error

**Shape**:
```
Test Error
     |
     |    ___
     |   /   \___
     |  /        \___
     | /            \___________
     |/                         
     +-------------------------> Model Complexity
       Under-   Interpolation   Over-
       param.   threshold       param.
```

**Phases**:
1. **Underparameterized** ($|\theta| < N$): More parameters improve generalization
2. **Interpolation threshold** ($|\theta| = N$): Peak error. Network barely fits data, memorizes noise.
3. **Overparameterized** ($|\theta| \gg N$): More parameters *improve* generalization! Network has freedom to find simpler solutions.

### Why Does Overparameterization Help?

**Implicit regularization**: Gradient descent on overparameterized networks finds solutions that:
- Fit training data perfectly (zero training error)
- Have minimal norm (implicit regularization)
- Generalize well

**Intuition**: With many parameters, multiple solutions fit the data. SGD's implicit bias steers toward "simple" solutions (low-frequency functions, smooth decision boundaries).

**Mathematical insight** (Neyshabur et al., 2017): For overparameterized linear networks:
$$\min_{\mathbf{W}} \|\mathbf{y} - \mathbf{W}\mathbf{X}\|^2$$

Gradient descent finds the minimum norm solution:
$$\mathbf{W}^* = \mathbf{y}\mathbf{X}^T(\mathbf{X}\mathbf{X}^T)^{-1}$$

This generalizes better than arbitrary solutions to the training set.

### Connection to Our Project

Our model has millions of parameters, far exceeding the training set size. Classical theory suggests massive overfitting, yet we achieve good test performance. Double descent explains this: We're in the overparameterized regime where excess capacity improves generalization.

**Practical implication**: Don't fear large models! Overparameterization + proper regularization (dropout, batch norm, data augmentation) leads to better generalization.

## Neural Tangent Kernels (NTK)*

**Recent theory** (Jacot et al., 2018): In the infinite-width limit, neural network training dynamics are equivalent to kernel regression with a fixed kernel.

### Intuition

For a network $f(\mathbf{x}; \theta)$, the **Neural Tangent Kernel** measures how parameter changes affect outputs:

$$K_{\text{NTK}}(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x}; \theta_0)^T \nabla_\theta f(\mathbf{x}'; \theta_0)$$

**Key result**: For infinitely wide networks, $K_{\text{NTK}}$ remains constant during training!

### Training Dynamics in NTK Regime

The network evolves according to:
$$f_t(\mathbf{x}) = f_0(\mathbf{x}) - \eta \int_0^t \sum_i K_{\text{NTK}}(\mathbf{x}, \mathbf{x}_i) (f_s(\mathbf{x}_i) - y_i) ds$$

This is kernel gradient descent with $K_{\text{NTK}}$.

**Implications**:
1. **Convergence guarantees**: Kernel methods have well-understood convergence properties
2. **Feature learning**: In NTK regime, features don't change (kernel fixed). No representation learning!
3. **Finite-width networks**: Deviate from NTK regime, enabling feature learning

### Lazy vs. Rich Training Regimes

**Lazy regime** (wide networks, large init): Parameters barely change, features static. Training is kernel regression.

**Rich regime** (finite width, small init): Parameters change significantly, features evolve. True representation learning occurs.

**Our model**: With 512 and 256 hidden units, we're in the rich regime. Features adapt during training, which is why transfer learning and fine-tuning are effective.

## Implicit Regularization of Gradient Descent*

**Question**: Why does gradient descent generalize well without explicit regularization?

**Answer**: SGD has implicit biases that favor simpler solutions.

### Implicit Bias Toward Low-Norm Solutions

For underparameterized linear regression, gradient descent converges to the minimum-norm solution:

$$\min_{\mathbf{w}} \|\mathbf{w}\|^2 \quad \text{subject to} \quad \mathbf{y} = \mathbf{X}\mathbf{w}$$

**Proof sketch**:
- Initialize $\mathbf{w}_0 = \mathbf{0}$
- Gradient descent updates: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{X}^T(\mathbf{X}\mathbf{w}_t - \mathbf{y})$
- Solution lies in span of $\mathbf{X}^T$: $\mathbf{w}_t = \mathbf{X}^T \mathbf{v}_t$ for some $\mathbf{v}_t$
- Converges to $\mathbf{w}^* = \mathbf{X}^T(\mathbf{X}\mathbf{X}^T)^{-1}\mathbf{y}$ (minimum norm)

### Implicit Bias in Deep Networks

For deep networks, SGD favors:
1. **Low-frequency functions**: Smooth decision boundaries over jagged ones
2. **Large margin classifiers**: Maximizes distance to decision boundary
3. **Sparse representations**: Many neurons inactive (ReLU outputs zero)

**Intuition**: Among the many functions that fit training data, SGD naturally selects simpler ones that generalize better.

### Edge of Stability

**Observation** (Cohen et al., 2021): Training often operates at learning rates beyond the traditional stability threshold.

**Classical theory**: For convergence, require $\eta < \frac{2}{\lambda_{\max}(H)}$ where $H$ is the Hessian's largest eigenvalue.

**Empirical reality**: Training is stable even when $\eta \cdot \lambda_{\max}(H) > 2$. The network operates at the "edge of stability," oscillating around minima but never diverging.

**Implication**: Aggressive learning rates can speed convergence without destroying stability, thanks to implicit regularization.

## Alternative Training Paradigms*

### Contrastive Learning

**Idea**: Learn representations by pulling similar examples together and pushing dissimilar examples apart.

**Loss** (SimCLR):
$$L = -\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_k) / \tau)}$$

where $(\mathbf{x}_i, \mathbf{x}_j)$ are augmented versions of the same image.

**Application to medical imaging**: Could pre-train on unlabeled MRI scans, then fine-tune on labeled pathology data.

### Self-Supervised Learning

**Idea**: Create pretext tasks from unlabeled data.

**Examples**:
- **Rotation prediction**: Rotate image by 0°, 90°, 180°, 270°, predict rotation
- **Inpainting**: Mask part of image, predict missing region
- **Context prediction**: Given image patch, predict location of neighboring patches

**Connection to our project**: Could leverage vast amounts of unlabeled brain MRIs to pre-train better features than ImageNet.

### Meta-Learning

**Idea**: Learn to learn. Train on many tasks, adapt quickly to new tasks with few examples.

**Algorithm** (MAML - Model-Agnostic Meta-Learning):
1. Sample tasks $\mathcal{T}_i$
2. For each task, compute gradient: $\theta'_i = \theta - \eta \nabla L_{\mathcal{T}_i}(\theta)$
3. Meta-update: $\theta \leftarrow \theta - \beta \nabla_\theta \sum_i L_{\mathcal{T}_i}(\theta'_i)$

**Application**: With few-shot pathology examples, meta-learning could enable rapid adaptation to rare tumor types.

## Connections to Kernel Methods*

### Support Vector Machines (SVMs)

SVMs find maximum-margin hyperplanes:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$$

**Kernel trick**: Replace inner products $\mathbf{x}_i^T \mathbf{x}_j$ with $k(\mathbf{x}_i, \mathbf{x}_j)$, enabling nonlinear boundaries.

**Connection to neural networks**: The final layer of our classifier is effectively a linear SVM on learned features $\mathbf{h}$. The network learns the kernel transformation.

### Gaussian Processes (GPs)

**Bayesian perspective**: Place prior over functions, update with data.

**Connection**: Infinitely wide single-layer networks with certain activations converge to GPs (Neal, 1996). Multi-layer networks have more complex GP limits.

**Advantage of NNs**: Finite-width networks escape GP behavior, enabling feature learning. GPs offer uncertainty quantification (useful for medical applications).

## Future Research Directions*

### Interpretability and Explainability

**Challenge**: Deep networks are "black boxes." For medical applications, we need explainable predictions.

**Approaches**:
- **Grad-CAM**: Visualize which image regions influence predictions
- **Attention mechanisms**: Explicitly weight important features
- **Concept activation vectors**: Detect presence of human-understandable concepts

**Next session**: We'll cover these in detail (Session 08: Interpretability).

### Uncertainty Quantification

**Challenge**: Classifiers output probabilities, but these aren't calibrated. A 90% prediction might be overconfident.

**Approaches**:
- **Ensembles**: Average predictions from multiple models
- **Monte Carlo Dropout**: Sample dropout masks at test time, estimate variance
- **Bayesian neural networks**: Maintain posterior distributions over weights

**Medical importance**: Knowing *when* the model is uncertain is crucial for safety.

### Fairness and Bias

**Challenge**: If training data overrepresents certain demographics, the model may perform poorly on underrepresented groups.

**Approaches**:
- **Balanced datasets**: Ensure diverse representation
- **Fairness metrics**: Measure performance across subgroups
- **Adversarial debiasing**: Train network to be invariant to sensitive attributes

**Session 10**: Ethics in Medical AI will cover these issues comprehensively.

### Continual Learning

**Challenge**: Models trained on fixed datasets can't adapt to new tumor types without forgetting old ones ("catastrophic forgetting").

**Approaches**:
- **Elastic Weight Consolidation**: Constrain important weights from changing
- **Progressive Neural Networks**: Add new columns for new tasks
- **Memory replay**: Interleave old and new examples during training

**Application**: As new pathologies are discovered, adapt the classifier without retraining from scratch.

### Automated Architecture Search (NAS)

**Idea**: Use algorithms to discover optimal architectures rather than hand-designing.

**Methods**:
- **Reinforcement learning**: Train RL agent to propose architectures
- **Evolutionary algorithms**: Mutate and select architectures
- **Gradient-based**: Differentiable architecture search (DARTS)

**Trade-off**: NAS requires enormous computational resources but can find architectures humans wouldn't design.

---

## Conclusion of Part F

These advanced topics illustrate the richness of deep learning theory and practice. While not all are immediately necessary for understanding our neuropathology classifier, they provide:

1. **Theoretical depth**: Understanding why neural networks work, not just how
2. **Research connections**: Links to cutting-edge papers and ongoing questions
3. **Practical insights**: Ideas for improving and extending the classifier
4. **Future-proofing**: Knowledge to adapt as the field evolves

As you progress through the tutorial, revisit these topics. Concepts that seem abstract initially will become concrete as you gain experience implementing and training networks.

**Next**: Session 02 will dive into CNNs and image processing, building on this foundation to understand the MobileNetV2 architecture in detail.

---

*Part F is now complete with comprehensive coverage of advanced topics including loss landscapes, representation learning, lottery tickets, double descent, NTK theory, implicit regularization, alternative training paradigms, and future research directions. All topics are rigorously explained with mathematical foundations.*
