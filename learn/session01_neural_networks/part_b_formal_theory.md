# Session 01 Part B: Neural Networks - Formal Theory and Definitions

## Table of Contents

1. [Mathematical Preliminaries](#mathematical-preliminaries)
2. [Formal Definition of Feedforward Neural Networks](#formal-definition-of-feedforward-neural-networks)
3. [Activation Functions: Properties and Choices](#activation-functions-properties-and-choices)
4. [Forward Propagation: Computing Network Output](#forward-propagation-computing-network-output)
5. [The Learning Problem: Empirical Risk Minimization](#the-learning-problem-empirical-risk-minimization)
6. [Loss Functions for Classification](#loss-functions-for-classification)
7. [The Softmax Function and Probability Interpretation](#the-softmax-function-and-probability-interpretation)
8. [Parameter Space and Optimization Landscape](#parameter-space-and-optimization-landscape)
9. [Generalization and the Bias-Variance Tradeoff](#generalization-and-the-bias-variance-tradeoff)

---

## Mathematical Preliminaries

Before diving into the formal definition of neural networks, we'll establish the mathematical notation and review essential concepts from linear algebra, calculus, and probability theory.

### Vectors and Matrices

A **vector** $\mathbf{x} \in \mathbb{R}^n$ is an ordered tuple of $n$ real numbers. We use boldface lowercase letters for vectors and typically treat them as column vectors:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

The **transpose** $\mathbf{x}^T$ is a row vector: $\mathbf{x}^T = [x_1, x_2, \ldots, x_n]$.

A **matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a rectangular array of numbers with $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

We use boldface uppercase letters for matrices.

The **inner product** (or dot product) of two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ is:

$$\mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

The **matrix-vector product** $\mathbf{A}\mathbf{x}$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$ produces a vector $\mathbf{y} \in \mathbb{R}^m$ where:

$$y_i = \sum_{j=1}^{n} a_{ij} x_j$$

This can be viewed as computing the inner product of each row of $\mathbf{A}$ with $\mathbf{x}$.

The **Euclidean norm** (or $L^2$ norm) of a vector $\mathbf{x}$ is:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$$

When we write $\|\mathbf{x}\|$ without subscript, we mean the $L^2$ norm by default.

### Functions and Composition

A **function** $f: \mathbb{R}^n \to \mathbb{R}^m$ maps vectors in $\mathbb{R}^n$ to vectors in $\mathbb{R}^m$. We write $\mathbf{y} = f(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{y} \in \mathbb{R}^m$.

**Composition** of functions $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^k$ produces a function $h = g \circ f: \mathbb{R}^n \to \mathbb{R}^k$ defined by:

$$h(\mathbf{x}) = g(f(\mathbf{x}))$$

Neural networks are built through repeated function composition.

An **affine function** has the form:

$$f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{m \times n}$ is a weight matrix and $\mathbf{b} \in \mathbb{R}^m$ is a bias vector. Affine functions combine a linear transformation ($\mathbf{W}\mathbf{x}$) with a translation ($\mathbf{b}$).

### Element-wise Operations

When we write $\sigma(\mathbf{z})$ for a vector $\mathbf{z} = [z_1, z_2, \ldots, z_n]^T$ and scalar function $\sigma: \mathbb{R} \to \mathbb{R}$, we mean **element-wise application**:

$$\sigma(\mathbf{z}) = \begin{bmatrix} \sigma(z_1) \\ \sigma(z_2) \\ \vdots \\ \sigma(z_n) \end{bmatrix}$$

Similarly, for element-wise multiplication (Hadamard product) of two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$:

$$\mathbf{x} \odot \mathbf{y} = \begin{bmatrix} x_1 y_1 \\ x_2 y_2 \\ \vdots \\ x_n y_n \end{bmatrix}$$

### Probability and Expectation

A **probability distribution** over a discrete set $\mathcal{Y} = \{1, 2, \ldots, K\}$ assigns a probability $p(y)$ to each element $y \in \mathcal{Y}$ such that:

$$p(y) \geq 0 \quad \text{for all } y, \quad \text{and} \quad \sum_{y \in \mathcal{Y}} p(y) = 1$$

The **expectation** of a function $f$ under distribution $p$ is:

$$\mathbb{E}_{y \sim p}[f(y)] = \sum_{y \in \mathcal{Y}} p(y) f(y)$$

For continuous distributions with density $p(x)$, the expectation is:

$$\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) \, dx$$

### Calculus: Gradients and Chain Rule

The **derivative** of a function $f: \mathbb{R} \to \mathbb{R}$ at point $x$ is:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

For a function $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is a vector of partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} 
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}$$

The gradient points in the direction of steepest ascent of $f$.

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** is:

$$\mathbf{J}_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

The **chain rule** for composite functions is fundamental to backpropagation. If $h = g \circ f$ where $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^k$, then:

$$\mathbf{J}_h(\mathbf{x}) = \mathbf{J}_g(f(\mathbf{x})) \cdot \mathbf{J}_f(\mathbf{x})$$

For scalar-valued functions, if $z = g(y)$ and $y = f(x)$, then:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

We'll use the chain rule extensively when deriving backpropagation.

## Formal Definition of Feedforward Neural Networks

A **feedforward neural network** (also called a multilayer perceptron or MLP) is a function $F: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$ constructed by composing $L$ layers of affine transformations and element-wise nonlinearities.

For layer $\ell = 1, 2, \ldots, L$, define:
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$: weight matrix for layer $\ell$
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$: bias vector for layer $\ell$
- $\sigma^{(\ell)}: \mathbb{R} \to \mathbb{R}$: activation function for layer $\ell$
- $n_\ell$: number of neurons (dimensionality) in layer $\ell$

The **layer-wise transformation** is:

$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$$

where:
- $\mathbf{z}^{(\ell)} \in \mathbb{R}^{n_\ell}$ is the **pre-activation** (or logit)
- $\mathbf{a}^{(\ell)} \in \mathbb{R}^{n_\ell}$ is the **activation** (or output) of layer $\ell$

The input is $\mathbf{a}^{(0)} = \mathbf{x} \in \mathbb{R}^{n_0}$, and the output is $\mathbf{a}^{(L)} = F(\mathbf{x}) \in \mathbb{R}^{n_L}$.

Unrolling the recursion, we can write:

$$F(\mathbf{x}) = \sigma^{(L)}\left(\mathbf{W}^{(L)} \sigma^{(L-1)}\left(\mathbf{W}^{(L-1)} \cdots \sigma^{(1)}\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \cdots + \mathbf{b}^{(L-1)}\right) + \mathbf{b}^{(L)}\right)$$

This is a composition of $L$ affine transformations and nonlinearities.

The **parameters** (or weights) of the network are:

$$\boldsymbol{\theta} = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\}$$

The total number of parameters is:

$$|\boldsymbol{\theta}| = \sum_{\ell=1}^{L} (n_\ell \cdot n_{\ell-1} + n_\ell)$$

For example, a network with architecture $[784, 128, 64, 10]$ (input size 784, two hidden layers with 128 and 64 neurons, output size 10) has:

$$(784 \times 128 + 128) + (128 \times 64 + 64) + (64 \times 10 + 10) = 100{,}480 + 8{,}256 + 650 = 109{,}386 \text{ parameters}$$

This is a remarkably compact representation of a function from $\mathbb{R}^{784}$ to $\mathbb{R}^{10}$.

### Terminology

- **Input layer**: Layer 0, containing the raw input $\mathbf{x}$.
- **Hidden layers**: Layers $1$ through $L-1$. These are called "hidden" because their values are not directly observed in the data; they are internal representations learned by the network.
- **Output layer**: Layer $L$, producing the final output $\mathbf{a}^{(L)}$.
- **Depth**: The number of layers $L$. "Deep" networks have many layers (typically $L \geq 3$).
- **Width**: The number of neurons in each layer. Wide networks have large $n_\ell$.
- **Architecture**: The specification of $(n_0, n_1, \ldots, n_L)$ and the choice of activation functions.

### Vectorization and Matrix Notation

When processing a batch of $B$ inputs $\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(B)}\}$, we stack them into a matrix $\mathbf{X} \in \mathbb{R}^{B \times n_0}$ where each row is an input:

$$\mathbf{X} = \begin{bmatrix}
(\mathbf{x}^{(1)})^T \\
(\mathbf{x}^{(2)})^T \\
\vdots \\
(\mathbf{x}^{(B)})^T
\end{bmatrix}$$

The layer transformation becomes:

$$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)} (\mathbf{W}^{(\ell)})^T + \mathbf{1}_B (\mathbf{b}^{(\ell)})^T$$
$$\mathbf{A}^{(\ell)} = \sigma^{(\ell)}(\mathbf{Z}^{(\ell)})$$

where $\mathbf{A}^{(\ell)} \in \mathbb{R}^{B \times n_\ell}$ and $\mathbf{1}_B \in \mathbb{R}^B$ is a vector of ones (used for broadcasting the bias).

In practice, deep learning frameworks handle this broadcasting automatically, and we write simply $\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)} (\mathbf{W}^{(\ell)})^T + \mathbf{b}^{(\ell)}$ with the understanding that the bias is added to each row.

This batched computation is crucial for computational efficiency, as it allows us to exploit parallelism in GPUs and vectorized CPU operations.

## Activation Functions: Properties and Choices

The activation function $\sigma$ introduces nonlinearity into the network. The choice of activation function has a significant impact on training dynamics and model performance.

### Desirable Properties

An ideal activation function should have several properties:

1. **Nonlinearity**: $\sigma$ is not linear, so the network can represent nonlinear functions.
2. **Differentiability**: $\sigma$ is differentiable almost everywhere, enabling gradient-based optimization.
3. **Monotonicity**: $\sigma$ is monotonic (either non-decreasing or non-increasing), which can help with optimization.
4. **Bounded or unbounded range**: Bounded range (e.g., $(0, 1)$) can help with gradient stability but may cause saturation; unbounded range (e.g., $[0, \infty)$) avoids saturation but may lead to exploding activations.
5. **Zero-centered**: Outputs centered around zero can improve training dynamics.
6. **Computational efficiency**: $\sigma$ and its derivative should be cheap to compute.

No single activation function is optimal for all tasks. The choice depends on the problem, network architecture, and training procedure.

### Common Activation Functions

#### Sigmoid (Logistic)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative**:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Properties**:
- Range: $(0, 1)$
- Smooth, monotonic, bounded
- Outputs can be interpreted as probabilities
- Saturates for large $|z|$: $\sigma(z) \to 0$ as $z \to -\infty$ and $\sigma(z) \to 1$ as $z \to \infty$

**Issues**:
- **Vanishing gradients**: When $|z|$ is large, $\sigma'(z) \approx 0$, causing gradients to vanish during backpropagation in deep networks.
- **Not zero-centered**: Outputs are always positive, which can slow convergence.
- **Computationally expensive**: Requires computing exponentials.

Sigmoid was popular in early neural networks but has largely been replaced by ReLU in hidden layers. It's still used in the output layer for binary classification.

#### Hyperbolic Tangent (Tanh)

$$\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}$$

**Derivative**:

$$\sigma'(z) = 1 - \tanh^2(z)$$

**Properties**:
- Range: $(-1, 1)$
- Smooth, monotonic, bounded
- Zero-centered (unlike sigmoid)
- Related to sigmoid: $\tanh(z) = 2\sigma(2z) - 1$

**Issues**:
- **Vanishing gradients**: Like sigmoid, saturates for large $|z|$.
- **Computationally expensive**: Requires computing exponentials.

Tanh is preferable to sigmoid for hidden layers because it's zero-centered, but it still suffers from vanishing gradients.

#### Rectified Linear Unit (ReLU)

$$\sigma(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Derivative**:

$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

(The derivative at $z = 0$ is undefined, but in practice we set it to 0 or 1; this doesn't matter because the probability of encountering exactly $z = 0$ is negligible.)

**Properties**:
- Range: $[0, \infty)$
- Non-smooth (not differentiable at $z = 0$), but piecewise linear
- Unbounded above
- Sparse activation: For any input, approximately half of the neurons output zero (assuming zero-centered data)

**Advantages**:
- **No vanishing gradient for positive values**: $\sigma'(z) = 1$ for $z > 0$, so gradients flow without attenuation.
- **Computationally efficient**: Just a comparison and multiplication.
- **Empirically effective**: ReLU and its variants are the default choice in modern deep learning.

**Issues**:
- **Dying ReLU**: If a neuron's weights are such that $z < 0$ for all inputs, the neuron outputs zero and its gradient is zero, so it never updates. This can happen if the learning rate is too high or if the network initializes poorly.
- **Not zero-centered**: Outputs are always non-negative.

Despite these issues, ReLU is the most widely used activation function in hidden layers of deep networks.

#### Leaky ReLU

To address the dying ReLU problem, Leaky ReLU introduces a small slope for negative values:

$$\sigma(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

where $\alpha$ is a small positive constant (e.g., $\alpha = 0.01$).

**Derivative**:

$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

Leaky ReLU ensures that neurons never completely die, as the gradient is always non-zero.

**Parametric ReLU (PReLU)** treats $\alpha$ as a learnable parameter, allowing the network to adapt the slope.

#### Exponential Linear Unit (ELU)

$$\sigma(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

**Derivative**:

$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \sigma(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

ELU is smooth and has negative values, which can help make the mean activation closer to zero. However, it requires computing exponentials for negative values, making it more expensive than ReLU.

#### Softmax

Softmax is used in the output layer for multi-class classification. Unlike the above element-wise activations, softmax is a vector-valued function:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

for $k = 1, 2, \ldots, K$.

**Properties**:
- Outputs sum to 1: $\sum_{k=1}^{K} \text{softmax}(\mathbf{z})_k = 1$
- Each output is in $(0, 1)$
- Outputs can be interpreted as class probabilities

Softmax converts logits (unnormalized scores) into a probability distribution. We'll derive its properties in detail in a later section.

### Activation Function Selection

For modern feedforward and convolutional networks, the standard choice is:
- **Hidden layers**: ReLU (or variants like Leaky ReLU, ELU)
- **Output layer for binary classification**: Sigmoid
- **Output layer for multi-class classification**: Softmax
- **Output layer for regression**: No activation (linear output)

In our brain tumor classifier, we use ReLU in the custom dense layers (the classification head) and softmax in the output layer to produce a 17-way probability distribution.

## Forward Propagation: Computing Network Output

Given an input $\mathbf{x}$, **forward propagation** computes the output $\mathbf{a}^{(L)}$ by sequentially applying each layer's transformation.

**Algorithm: Forward Propagation**

**Input**: $\mathbf{x} \in \mathbb{R}^{n_0}$, parameters $\boldsymbol{\theta} = \{\mathbf{W}^{(\ell)}, \mathbf{b}^{(\ell)}\}_{\ell=1}^{L}$

**Output**: $\mathbf{a}^{(L)} \in \mathbb{R}^{n_L}$

1. Set $\mathbf{a}^{(0)} = \mathbf{x}$
2. For $\ell = 1$ to $L$:
   - Compute $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$
   - Compute $\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$
3. Return $\mathbf{a}^{(L)}$

This is a straightforward computation that proceeds layer by layer from input to output. The complexity is dominated by the matrix-vector multiplications, which are $O(n_\ell \cdot n_{\ell-1})$ for each layer.

For a batch of $B$ inputs, we use the batched version, processing all inputs in parallel. The complexity becomes $O(B \cdot n_\ell \cdot n_{\ell-1})$ per layer, but the constant factor is much smaller due to vectorization.

### Computational Graph

We can represent the forward pass as a **computational graph**, a directed acyclic graph (DAG) where nodes are variables and edges represent operations. For a simple two-layer network with one hidden layer:

```
x → [W1, b1] → z1 → σ → a1 → [W2, b2] → z2 → σ → a2 (output)
```

Each arrow represents a computation. The computational graph is useful for understanding the flow of information and, crucially, for deriving backpropagation via automatic differentiation.

### Shape Tracking

It's essential to keep track of tensor shapes to avoid dimensional mismatches. For layer $\ell$:
- Input: $\mathbf{a}^{(\ell-1)} \in \mathbb{R}^{n_{\ell-1}}$
- Weight matrix: $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$
- Bias: $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$
- Pre-activation: $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$
- Activation: $\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)}) \in \mathbb{R}^{n_\ell}$

For batched inputs $\mathbf{X} \in \mathbb{R}^{B \times n_0}$:
- $\mathbf{A}^{(\ell)} \in \mathbb{R}^{B \times n_\ell}$
- $\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)} (\mathbf{W}^{(\ell)})^T + \mathbf{b}^{(\ell)} \in \mathbb{R}^{B \times n_\ell}$

Deep learning frameworks (TensorFlow, PyTorch) automatically track shapes and perform broadcasting, but understanding the dimensions is crucial for debugging and designing architectures.

## The Learning Problem: Empirical Risk Minimization

Training a neural network means finding parameters $\boldsymbol{\theta}$ that make the network's predictions accurate on the training data and (hopefully) on new, unseen data.

### Supervised Learning Setup

We have a dataset of $N$ labeled examples:

$$\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$$

where $\mathbf{x}^{(i)} \in \mathbb{R}^{n_0}$ is an input (e.g., an image) and $y^{(i)} \in \mathcal{Y}$ is a label (e.g., one of $K$ classes).

For classification, $\mathcal{Y} = \{1, 2, \ldots, K\}$. For our brain tumor classifier, $K = 17$.

The **hypothesis class** is the set of functions our model can represent. For a neural network with architecture $(n_0, n_1, \ldots, n_L)$ and activation functions $\{\sigma^{(\ell)}\}$:

$$\mathcal{H} = \{F_{\boldsymbol{\theta}}: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L} \mid \boldsymbol{\theta} \in \Theta\}$$

where $\Theta$ is the parameter space.

Our goal is to find a function $f \in \mathcal{H}$ that minimizes the **expected risk** (or generalization error):

$$R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[\ell(f(\mathbf{x}), y)]$$

where $P$ is the true data distribution and $\ell$ is a loss function measuring the discrepancy between the prediction $f(\mathbf{x})$ and the true label $y$.

However, we don't know $P$; we only have the training data $\mathcal{D}$. So we approximate the expected risk with the **empirical risk** (or training loss):

$$\hat{R}(f) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(\mathbf{x}^{(i)}), y^{(i)})$$

**Empirical risk minimization (ERM)** is the principle of finding:

$$f^* = \arg\min_{f \in \mathcal{H}} \hat{R}(f)$$

For neural networks:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \frac{1}{N} \sum_{i=1}^{N} \ell(F_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)})$$

### The Optimization Problem

Training a neural network is an optimization problem: find the parameters $\boldsymbol{\theta}$ that minimize the empirical risk.

The objective function is:

$$J(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(F_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)})$$

We seek:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

This is a high-dimensional nonconvex optimization problem. The parameter space $\Theta \subset \mathbb{R}^d$ can have millions or billions of dimensions (for large networks).

Unlike convex optimization, there's no guarantee of finding the global minimum, or even a good local minimum. However, gradient-based methods (like stochastic gradient descent) have proven remarkably effective in practice.

### Regularization

To prevent overfitting (memorizing the training data), we often add a **regularization term** to the objective:

$$J_{\text{reg}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(F_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)}) + \lambda \Omega(\boldsymbol{\theta})$$

where $\Omega(\boldsymbol{\theta})$ is a penalty on complex models (e.g., $L^2$ weight decay: $\Omega(\boldsymbol{\theta}) = \sum_{\ell, i, j} (W_{ij}^{(\ell)})^2$) and $\lambda$ is a hyperparameter controlling the strength of regularization.

We'll discuss regularization techniques in detail in Session 05.

## Loss Functions for Classification

The loss function $\ell$ quantifies how wrong a prediction is. For classification, we need a loss that encourages the correct class to have high probability and incorrect classes to have low probability.

### 0-1 Loss

The most intuitive loss is the **0-1 loss**:

$$\ell_{0-1}(\hat{y}, y) = \mathbb{1}[\hat{y} \neq y] = \begin{cases} 0 & \text{if } \hat{y} = y \\ 1 & \text{if } \hat{y} \neq y \end{cases}$$

This simply counts classification errors. However, the 0-1 loss is not differentiable and doesn't provide gradient information, making it unsuitable for gradient-based optimization.

### Probabilistic Interpretation

Instead, we interpret the network's output as a probability distribution over classes. For an input $\mathbf{x}$, let $\mathbf{a}^{(L)} = [\hat{p}_1, \hat{p}_2, \ldots, \hat{p}_K]^T$ where $\hat{p}_k$ is the predicted probability of class $k$.

We use the **negative log-likelihood** (or cross-entropy) loss:

$$\ell(\hat{\mathbf{p}}, y) = -\log \hat{p}_y$$

If the true class is $y$, we want $\hat{p}_y$ to be large (close to 1). The loss $-\log \hat{p}_y$ is small when $\hat{p}_y$ is large and becomes infinite as $\hat{p}_y \to 0$, heavily penalizing confident wrong predictions.

### Cross-Entropy Loss

To derive the cross-entropy loss, we start with maximum likelihood estimation. Assume the data is generated i.i.d. from some distribution $P$, and our model defines a distribution $P_{\boldsymbol{\theta}}$ over labels given inputs.

The **likelihood** of the data under the model is:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} P_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$$

The **log-likelihood** is:

$$\log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \log P_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$$

Maximum likelihood estimation seeks to maximize the log-likelihood, equivalently minimize the negative log-likelihood:

$$-\log \mathcal{L}(\boldsymbol{\theta}) = -\sum_{i=1}^{N} \log P_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$$

Dividing by $N$:

$$J(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^{N} \log P_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$$

This is the **cross-entropy loss**. For each example $i$, the loss is $-\log P_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$, which is exactly the negative log-likelihood of the true class.

In practice, the network outputs logits $\mathbf{z}^{(L)}$, which are converted to probabilities via softmax:

$$\hat{p}_k = \frac{e^{z_k^{(L)}}}{\sum_{j=1}^{K} e^{z_j^{(L)}}}$$

The cross-entropy loss for example $i$ with true class $y^{(i)}$ is:

$$\ell_i = -\log \hat{p}_{y^{(i)}} = -\log \left( \frac{e^{z_{y^{(i)}}^{(L)}}}{\sum_{j=1}^{K} e^{z_j^{(L)}}} \right) = -z_{y^{(i)}}^{(L)} + \log \sum_{j=1}^{K} e^{z_j^{(L)}}$$

The total loss over the dataset is:

$$J(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell_i$$

### One-Hot Encoding

Often, labels are represented as **one-hot vectors**. For a $K$-class problem, the one-hot encoding of class $k$ is a vector $\mathbf{y} \in \{0, 1\}^K$ where $y_j = 1$ if $j = k$ and $y_j = 0$ otherwise.

For example, if $K = 3$ and the true class is 2, the one-hot vector is $[0, 1, 0]^T$.

Using one-hot encoding, the cross-entropy loss can be written as:

$$\ell = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

Since $y_k = 0$ for all $k$ except the true class, only the term corresponding to the true class contributes to the sum, recovering $-\log \hat{p}_{y_{\text{true}}}$.

The cross-entropy between two distributions $P$ and $Q$ is generally defined as:

$$H(P, Q) = -\sum_{k=1}^{K} P(k) \log Q(k)$$

where $P$ is the true distribution (one-hot in classification) and $Q$ is the predicted distribution (softmax output). This is why it's called "cross-entropy."

## The Softmax Function and Probability Interpretation

The **softmax function** converts a vector of real-valued logits into a probability distribution.

### Definition

Given a vector $\mathbf{z} = [z_1, z_2, \ldots, z_K]^T \in \mathbb{R}^K$, the softmax function is:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

for $k = 1, 2, \ldots, K$.

### Properties

1. **Outputs are positive**: Since $e^{z_k} > 0$ for all $z_k$, we have $\text{softmax}(\mathbf{z})_k > 0$.

2. **Outputs sum to 1**:

   $$\sum_{k=1}^{K} \text{softmax}(\mathbf{z})_k = \sum_{k=1}^{K} \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = \frac{\sum_{k=1}^{K} e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = 1$$

   Thus, the output is a valid probability distribution.

3. **Monotonic in each component**: If $z_k$ increases while other components remain fixed, $\text{softmax}(\mathbf{z})_k$ increases and $\text{softmax}(\mathbf{z})_j$ decreases for $j \neq k$.

4. **Translation invariant**: Adding a constant to all logits doesn't change the output:

   $$\text{softmax}(\mathbf{z} + c\mathbf{1})_k = \frac{e^{z_k + c}}{\sum_{j} e^{z_j + c}} = \frac{e^c e^{z_k}}{e^c \sum_{j} e^{z_j}} = \frac{e^{z_k}}{\sum_{j} e^{z_j}} = \text{softmax}(\mathbf{z})_k$$

   This property is used for numerical stability: we typically subtract $\max_k z_k$ before computing softmax to avoid overflow.

5. **Approaches one-hot as logits diverge**: If $z_k \gg z_j$ for all $j \neq k$, then $\text{softmax}(\mathbf{z})_k \approx 1$ and $\text{softmax}(\mathbf{z})_j \approx 0$ for $j \neq k$.

### Connection to Logistic Regression

For binary classification ($K = 2$), softmax reduces to logistic regression. Let $\mathbf{z} = [z_1, z_2]^T$. Then:

$$\text{softmax}(\mathbf{z})_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}$$

Dividing numerator and denominator by $e^{z_1}$:

$$\text{softmax}(\mathbf{z})_1 = \frac{1}{1 + e^{z_2 - z_1}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

where $\sigma$ is the sigmoid function. So softmax is a generalization of sigmoid to multiple classes.

### Softmax as Maximum Entropy

Softmax can be derived from the principle of maximum entropy. Given logits $\mathbf{z}$, we seek a distribution $\mathbf{p} = [p_1, \ldots, p_K]^T$ that:
- Respects the constraints: $p_k \geq 0$ and $\sum_k p_k = 1$
- Matches the expected logits: $\mathbb{E}_{k \sim p}[\mathbf{z}] = \sum_k p_k z_k$ equals a target value
- Maximizes entropy: $H(p) = -\sum_k p_k \log p_k$

The solution to this constrained optimization is the softmax distribution. This is known as the **Gibbs distribution** in statistical mechanics.

### Numerical Stability

Computing softmax naively can cause numerical overflow if the logits are large, since $e^{z_k}$ grows rapidly. To avoid this, we use the translation invariance property:

$$\text{softmax}(\mathbf{z})_k = \text{softmax}(\mathbf{z} - \max_j z_j)_k$$

By subtracting the maximum logit, all exponents become non-positive, preventing overflow. The largest exponent is $e^0 = 1$, and the others are smaller.

In code:

```python
def softmax(z):
    z_shifted = z - np.max(z)  # for numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

Deep learning frameworks handle this automatically.

## Parameter Space and Optimization Landscape

The parameter space $\Theta$ of a neural network is high-dimensional and the loss surface $J(\boldsymbol{\theta})$ is highly nonconvex. Understanding the geometry of this landscape is key to understanding training dynamics.

### Dimensionality

For a network with millions of parameters, $\boldsymbol{\theta} \in \mathbb{R}^d$ where $d \sim 10^6$ to $10^9$. Visualizing such a space is impossible, but we can reason about its properties.

### Nonconvexity

A function $f$ is **convex** if for any $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$$

Geometrically, the line segment between any two points on the graph lies above the graph.

Neural network loss functions are **not convex**. They have many local minima, saddle points, and plateaus. This makes optimization challenging, as gradient descent can get stuck in local minima or saddle points.

However, empirical evidence suggests that for overparameterized networks (where the number of parameters exceeds the number of training examples), most local minima have similar loss values close to the global minimum. Moreover, saddle points (where the gradient is zero but the Hessian has negative eigenvalues) are more common than local minima in high dimensions.

### Symmetries and Redundancy

The parameter space has extensive symmetries. For example, permuting the neurons in a hidden layer produces an identical function:

If we permute the rows of $\mathbf{W}^{(\ell)}$ and the columns of $\mathbf{W}^{(\ell+1)}$ consistently, the function $F_{\boldsymbol{\theta}}$ doesn't change. A network with $n_\ell$ neurons in layer $\ell$ has $n_\ell!$ equivalent parameterizations.

This means the parameter space contains many equivalent solutions. The loss surface has flat valleys where infinitely many parameter settings yield the same function.

### Initialization Matters

Because the loss is nonconvex, the initialization of $\boldsymbol{\theta}$ affects the final solution. Random initialization (with appropriate variance) is crucial. We'll discuss initialization strategies (Xavier, He initialization) in Session 04.

### The Role of Overparameterization

Recent theoretical work suggests that overparameterization (having more parameters than training examples) helps optimization. In the "lazy training" regime, an overparameterized network behaves like a linear model, and gradient descent converges to a global minimum. In practice, overparameterization also improves generalization, contrary to classical statistical learning theory—a phenomenon sometimes called the "double descent" curve.

## Generalization and the Bias-Variance Tradeoff

The ultimate goal of training is not to minimize training loss but to achieve low **test loss** (or generalization error) on new, unseen data.

### Train vs. Test Performance

- **Training error**: $\hat{R}(f) = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \ell(f(\mathbf{x}^{(i)}), y^{(i)})$ on the training set.
- **Test error**: $R_{\text{test}}(f) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \ell(f(\tilde{\mathbf{x}}^{(i)}), \tilde{y}^{(i)})$ on a held-out test set.

We want low test error. However, optimization directly minimizes training error. The difference between test and training error is the **generalization gap**.

### Overfitting and Underfitting

- **Underfitting**: The model is too simple to capture the underlying patterns. Both training and test errors are high.
- **Overfitting**: The model memorizes the training data, including noise. Training error is low, but test error is high.

The goal is to find a model complexity that balances these extremes.

### Bias-Variance Tradeoff

The expected test error of a model can be decomposed into three components:

$$\mathbb{E}[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias**: Error from incorrect assumptions in the model (e.g., assuming a linear model for nonlinear data). High bias leads to underfitting.
- **Variance**: Error from sensitivity to small fluctuations in the training set. High variance leads to overfitting.
- **Irreducible error**: Noise in the data that no model can eliminate.

Simpler models have high bias and low variance. Complex models have low bias and high variance. The tradeoff is to find the right complexity.

For neural networks, model complexity is controlled by:
- **Architecture**: Number of layers, neurons per layer.
- **Regularization**: Dropout, weight decay, early stopping.
- **Training time**: Training longer can lead to overfitting.

### Generalization in Deep Learning

Classical learning theory predicts that large models should overfit. However, deep neural networks often generalize well despite having millions of parameters. This is an active area of research.

Possible explanations include:
- **Implicit regularization**: Gradient descent implicitly prefers simpler functions (in some sense).
- **Overparameterization helps**: Having more parameters than data can improve both optimization and generalization.
- **Architecture induces inductive biases**: Convolutional networks, for instance, assume translation invariance, which is appropriate for images.

### Validation Sets

To tune hyperparameters (learning rate, regularization strength, architecture choices) without overfitting to the test set, we use a **validation set**.

The typical split is:
- **Training set**: Used to optimize $\boldsymbol{\theta}$.
- **Validation set**: Used to select hyperparameters and monitor generalization.
- **Test set**: Used only once at the end to report final performance.

Never train on the test set, and avoid excessive tuning on the validation set (which can lead to indirect overfitting).

### Cross-Validation

When data is scarce, **k-fold cross-validation** can be used:
1. Split data into $k$ folds.
2. For each fold $i$, train on the other $k-1$ folds and validate on fold $i$.
3. Average the validation errors across folds.

This provides a more reliable estimate of generalization but is computationally expensive (requires training $k$ models).

For deep learning with large datasets, a single train/validation/test split is usually sufficient.

---

In this part, we've established the formal mathematical framework for feedforward neural networks, defined key concepts like activation functions, forward propagation, loss functions, and the optimization problem. We've also discussed the challenge of generalization and the bias-variance tradeoff.

In the next part, we'll derive the mathematics of gradient computation and backpropagation, the algorithm that makes training deep networks feasible.
