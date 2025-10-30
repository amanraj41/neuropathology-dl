# Session 01 Part C: Neural Networks - Mathematical Derivations

## Table of Contents

1. [Calculus Review: Derivatives and the Chain Rule](#calculus-review-derivatives-and-the-chain-rule)
2. [Gradient Computation for Simple Networks](#gradient-computation-for-simple-networks)
3. [Backpropagation: The General Algorithm](#backpropagation-the-general-algorithm)
4. [Derivation of Backpropagation from First Principles](#derivation-of-backpropagation-from-first-principles)
5. [Computing Gradients for Common Activation Functions](#computing-gradients-for-common-activation-functions)
6. [Softmax and Cross-Entropy: Combined Gradient](#softmax-and-cross-entropy-combined-gradient)
7. [Gradient Flow and Vanishing/Exploding Gradients](#gradient-flow-and-vanishing-exploding-gradients)
8. [Matrix Calculus and Vectorization](#matrix-calculus-and-vectorization)
9. [Computational Complexity of Backpropagation](#computational-complexity-of-backpropagation)

---

## Calculus Review: Derivatives and the Chain Rule

Before deriving backpropagation, we'll review essential calculus concepts. Understanding these thoroughly is crucial for grasping gradient-based learning.

### Single-Variable Derivatives

The **derivative** of a function $f: \mathbb{R} \to \mathbb{R}$ measures how $f$ changes with respect to its input. Formally:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

The derivative $f'(x)$ represents the slope of the tangent line to $f$ at point $x$. If $f'(x) > 0$, the function is increasing; if $f'(x) < 0$, it's decreasing.

**Common derivatives:**

1. **Power rule**: If $f(x) = x^n$, then $f'(x) = nx^{n-1}$

   *Proof for $n = 2$*:
   $$\frac{d}{dx}(x^2) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} \frac{2xh + h^2}{h} = \lim_{h \to 0} (2x + h) = 2x$$

2. **Exponential**: If $f(x) = e^x$, then $f'(x) = e^x$

3. **Logarithm**: If $f(x) = \ln x$, then $f'(x) = \frac{1}{x}$

4. **Constant rule**: If $f(x) = c$ (constant), then $f'(x) = 0$

5. **Constant multiple**: If $f(x) = cg(x)$, then $f'(x) = cg'(x)$

6. **Sum rule**: If $f(x) = g(x) + h(x)$, then $f'(x) = g'(x) + h'(x)$

### The Chain Rule

The **chain rule** is the cornerstone of backpropagation. For composite functions, it tells us how to compute derivatives.

**Single-variable chain rule**: If $y = g(u)$ and $u = f(x)$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

*Example*: Let $y = (3x + 5)^2$. We can write $y = u^2$ where $u = 3x + 5$.

Then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 2(3x + 5) \cdot 3 = 6(3x + 5) = 18x + 30$$

We can verify this by expanding: $y = (3x+5)^2 = 9x^2 + 30x + 25$, so $\frac{dy}{dx} = 18x + 30$. ✓

**Extended chain rule** for longer compositions: If $z = h(y)$, $y = g(u)$, $u = f(x)$, then:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{du} \cdot \frac{du}{dx}$$

This telescoping product is the key to backpropagation: gradients propagate backward through the network by multiplying local derivatives.

### Multivariable Derivatives

For functions $f: \mathbb{R}^n \to \mathbb{R}$ (scalar output, vector input), we compute **partial derivatives** with respect to each input:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

The **gradient** is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient points in the direction of steepest ascent. To minimize $f$, we move in the direction $-\nabla f$.

**Multivariable chain rule**: If $z = f(y_1, y_2, \ldots, y_m)$ and each $y_j = g_j(x_1, x_2, \ldots, x_n)$, then:

$$\frac{\partial z}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

This is the multivariate generalization we'll use repeatedly.

### Derivative of a Sum

If $z = \sum_{i=1}^{n} f_i(x)$, then:

$$\frac{dz}{dx} = \sum_{i=1}^{n} \frac{df_i}{dx}$$

This follows from linearity of differentiation. We use this when computing gradients of loss functions summed over training examples or network outputs.

## Gradient Computation for Simple Networks

Let's derive gradients for a simple two-layer network to build intuition before tackling the general case.

### Setup: Two-Layer Network

Consider a network with:
- Input: $\mathbf{x} \in \mathbb{R}^{n_0}$
- Hidden layer: $n_1$ neurons
- Output: $\mathbf{y} \in \mathbb{R}^{n_2}$

The forward pass computes:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \in \mathbb{R}^{n_1}$$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)}) \in \mathbb{R}^{n_1}$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)} \in \mathbb{R}^{n_2}$$\mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)}) \in \mathbb{R}^{n_2}$$

The loss for a single example is:

$$L = \ell(\mathbf{a}^{(2)}, \mathbf{y}_{\text{true}})$$

For concreteness, let's use mean squared error:

$$L = \frac{1}{2} \|\mathbf{a}^{(2)} - \mathbf{y}_{\text{true}}\|^2 = \frac{1}{2} \sum_{i=1}^{n_2} (a_i^{(2)} - y_i)^2$$

Our goal is to compute $\frac{\partial L}{\partial \mathbf{W}^{(1)}}$, $\frac{\partial L}{\partial \mathbf{b}^{(1)}}$, $\frac{\partial L}{\partial \mathbf{W}^{(2)}}$, $\frac{\partial L}{\partial \mathbf{b}^{(2)}}$.

### Output Layer Gradients

Start with the output layer. The loss depends directly on $\mathbf{a}^{(2)}$:

$$\frac{\partial L}{\partial a_i^{(2)}} = \frac{\partial}{\partial a_i^{(2)}} \left[ \frac{1}{2} \sum_{j=1}^{n_2} (a_j^{(2)} - y_j)^2 \right] = a_i^{(2)} - y_i$$

In vector form:

$$\frac{\partial L}{\partial \mathbf{a}^{(2)}} = \mathbf{a}^{(2)} - \mathbf{y}_{\text{true}}$$

This is the **error** at the output.

Next, compute $\frac{\partial L}{\partial \mathbf{z}^{(2)}}$ using the chain rule. Since $a_i^{(2)} = \sigma(z_i^{(2)})$:

$$\frac{\partial L}{\partial z_i^{(2)}} = \frac{\partial L}{\partial a_i^{(2)}} \cdot \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}} = (a_i^{(2)} - y_i) \cdot \sigma'(z_i^{(2)})$$

In vector form (using element-wise multiplication $\odot$):

$$\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \left( \mathbf{a}^{(2)} - \mathbf{y}_{\text{true}} \right) \odot \sigma'(\mathbf{z}^{(2)})$$

Define the **error signal**:

$$\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}}$$

Now compute gradients with respect to $\mathbf{W}^{(2)}$ and $\mathbf{b}^{(2)}$. Recall $z_i^{(2)} = \sum_{j=1}^{n_1} W_{ij}^{(2)} a_j^{(1)} + b_i^{(2)}$.

For the weight $W_{ij}^{(2)}$:

$$\frac{\partial L}{\partial W_{ij}^{(2)}} = \frac{\partial L}{\partial z_i^{(2)}} \cdot \frac{\partial z_i^{(2)}}{\partial W_{ij}^{(2)}} = \delta_i^{(2)} \cdot a_j^{(1)}$$

In matrix form:

$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T$$

This is an **outer product**: $\boldsymbol{\delta}^{(2)} \in \mathbb{R}^{n_2}$ and $\mathbf{a}^{(1)} \in \mathbb{R}^{n_1}$, so the result is $n_2 \times n_1$, matching the shape of $\mathbf{W}^{(2)}$.

For the bias:

$$\frac{\partial L}{\partial b_i^{(2)}} = \frac{\partial L}{\partial z_i^{(2)}} \cdot \frac{\partial z_i^{(2)}}{\partial b_i^{(2)}} = \delta_i^{(2)} \cdot 1 = \delta_i^{(2)}$$

In vector form:

$$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$$

### Hidden Layer Gradients

Now for the hidden layer. The challenge is that $L$ doesn't depend directly on $\mathbf{a}^{(1)}$; it depends on it through $\mathbf{z}^{(2)}$.

Using the chain rule:

$$\frac{\partial L}{\partial a_j^{(1)}} = \sum_{i=1}^{n_2} \frac{\partial L}{\partial z_i^{(2)}} \cdot \frac{\partial z_i^{(2)}}{\partial a_j^{(1)}}$$

Since $z_i^{(2)} = \sum_{k=1}^{n_1} W_{ik}^{(2)} a_k^{(1)} + b_i^{(2)}$, we have $\frac{\partial z_i^{(2)}}{\partial a_j^{(1)}} = W_{ij}^{(2)}$.

Thus:

$$\frac{\partial L}{\partial a_j^{(1)}} = \sum_{i=1}^{n_2} \delta_i^{(2)} \cdot W_{ij}^{(2)}$$

In matrix form:

$$\frac{\partial L}{\partial \mathbf{a}^{(1)}} = (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}$$

This is the key step: the error signal **propagates backward** through the transpose of the weight matrix.

Next, compute $\frac{\partial L}{\partial \mathbf{z}^{(1)}}$:

$$\frac{\partial L}{\partial z_j^{(1)}} = \frac{\partial L}{\partial a_j^{(1)}} \cdot \frac{\partial a_j^{(1)}}{\partial z_j^{(1)}} = \frac{\partial L}{\partial a_j^{(1)}} \cdot \sigma'(z_j^{(1)})$$

Define:

$$\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left[ (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)} \right] \odot \sigma'(\mathbf{z}^{(1)})$$

Finally, compute gradients with respect to $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$:

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

### Summary of the Two-Layer Derivation

**Forward pass:**
1. $\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$
2. $\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$
3. $\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$
4. $\mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)})$
5. $L = \ell(\mathbf{a}^{(2)}, \mathbf{y})$

**Backward pass:**
1. $\boldsymbol{\delta}^{(2)} = (\mathbf{a}^{(2)} - \mathbf{y}) \odot \sigma'(\mathbf{z}^{(2)})$
2. $\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T$
3. $\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$
4. $\boldsymbol{\delta}^{(1)} = [(\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}] \odot \sigma'(\mathbf{z}^{(1)})$
5. $\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$
6. $\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$

The pattern is clear: 
- Error signals $\boldsymbol{\delta}^{(\ell)}$ propagate backward.
- Weight gradients are outer products: $\boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$.
- Bias gradients are just $\boldsymbol{\delta}^{(\ell)}$.
- Errors propagate through $(\mathbf{W}^{(\ell)})^T$ and element-wise by $\sigma'(\mathbf{z}^{(\ell-1)})$.

## Backpropagation: The General Algorithm

Now we'll generalize to networks of arbitrary depth $L$.

### The Backpropagation Algorithm

**Input**: Training example $(\mathbf{x}, \mathbf{y})$, network parameters $\boldsymbol{\theta}$

**Output**: Gradients $\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}$, $\frac{\partial L}{\partial \mathbf{b}^{(\ell)}}$ for $\ell = 1, \ldots, L$

**Forward Pass:**

1. Set $\mathbf{a}^{(0)} = \mathbf{x}$
2. For $\ell = 1$ to $L$:
   - $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$
   - $\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$
3. Compute loss: $L = \ell(\mathbf{a}^{(L)}, \mathbf{y})$

**Backward Pass:**

1. Compute output error:
   $$\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot \sigma'^{(L)}(\mathbf{z}^{(L)})$$

2. For $\ell = L, L-1, \ldots, 1$:
   - Compute weight gradient:
     $$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$$
   - Compute bias gradient:
     $$\frac{\partial L}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$
   - If $\ell > 1$, propagate error backward:
     $$\boldsymbol{\delta}^{(\ell-1)} = [(\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}] \odot \sigma'^{(\ell-1)}(\mathbf{z}^{(\ell-1)})$$

3. Return all gradients

### Intuition: Flow of Information

In the **forward pass**, information flows from input to output:

$$\mathbf{x} \to \mathbf{z}^{(1)} \to \mathbf{a}^{(1)} \to \mathbf{z}^{(2)} \to \cdots \to \mathbf{a}^{(L)} \to L$$

In the **backward pass**, error information flows from output to input:

$$L \to \frac{\partial L}{\partial \mathbf{a}^{(L)}} \to \boldsymbol{\delta}^{(L)} \to \boldsymbol{\delta}^{(L-1)} \to \cdots \to \boldsymbol{\delta}^{(1)}$$

At each layer, we compute:
1. How the loss changes with respect to the layer's output ($\boldsymbol{\delta}^{(\ell)}$).
2. How this affects the layer's parameters ($\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}$, $\frac{\partial L}{\partial \mathbf{b}^{(\ell)}}$).
3. How to propagate the error to the previous layer.

This is called "backpropagation" because errors propagate backward through the network.

### Why It's Called "Backpropagation"

The algorithm was popularized by Rumelhart, Hinton, and Williams in their seminal 1986 paper "Learning representations by back-propagating errors." The key insight was that by applying the chain rule systematically, we can compute all gradients in a single backward pass through the network, with computational cost comparable to the forward pass.

Before backpropagation, training neural networks was computationally intractable for all but the smallest networks. Backpropagation made deep learning possible.

## Derivation of Backpropagation from First Principles

Let's rigorously derive backpropagation using the multivariate chain rule.

### Notation

For layer $\ell$:
- $\mathbf{a}^{(\ell)} \in \mathbb{R}^{n_\ell}$: activations
- $\mathbf{z}^{(\ell)} \in \mathbb{R}^{n_\ell}$: pre-activations
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$: weights
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$: biases
- $\sigma^{(\ell)}$: activation function

The forward pass computes:

$$z_i^{(\ell)} = \sum_{j=1}^{n_{\ell-1}} W_{ij}^{(\ell)} a_j^{(\ell-1)} + b_i^{(\ell)}$$a_i^{(\ell)} = \sigma^{(\ell)}(z_i^{(\ell)})$$

The loss $L$ depends on $\mathbf{a}^{(L)}$ and the true label $\mathbf{y}$.

### Gradient with Respect to Output Layer Pre-activations

Define:

$$\delta_i^{(L)} = \frac{\partial L}{\partial z_i^{(L)}}$$

By the chain rule:

$$\delta_i^{(L)} = \frac{\partial L}{\partial a_i^{(L)}} \cdot \frac{\partial a_i^{(L)}}{\partial z_i^{(L)}} = \frac{\partial L}{\partial a_i^{(L)}} \cdot \sigma'^{(L)}(z_i^{(L)})$$

In vector form:

$$\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot \sigma'^{(L)}(\mathbf{z}^{(L)})$$

### Gradient with Respect to Weights and Biases

For weight $W_{ij}^{(\ell)}$, the loss depends on it through $z_i^{(\ell)}$:

$$\frac{\partial L}{\partial W_{ij}^{(\ell)}} = \frac{\partial L}{\partial z_i^{(\ell)}} \cdot \frac{\partial z_i^{(\ell)}}{\partial W_{ij}^{(\ell)}}$$

Since $z_i^{(\ell)} = \sum_{k} W_{ik}^{(\ell)} a_k^{(\ell-1)} + b_i^{(\ell)}$, we have:

$$\frac{\partial z_i^{(\ell)}}{\partial W_{ij}^{(\ell)}} = a_j^{(\ell-1)}$$

Thus:

$$\frac{\partial L}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell)} \cdot a_j^{(\ell-1)}$$

In matrix form:

$$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$$

For bias $b_i^{(\ell)}$:

$$\frac{\partial L}{\partial b_i^{(\ell)}} = \frac{\partial L}{\partial z_i^{(\ell)}} \cdot \frac{\partial z_i^{(\ell)}}{\partial b_i^{(\ell)}} = \delta_i^{(\ell)} \cdot 1 = \delta_i^{(\ell)}$$

In vector form:

$$\frac{\partial L}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$

### Propagating Error to Previous Layer

For hidden layer $\ell-1$, we need $\boldsymbol{\delta}^{(\ell-1)} = \frac{\partial L}{\partial \mathbf{z}^{(\ell-1)}}$.

The loss depends on $z_j^{(\ell-1)}$ through $a_j^{(\ell-1)}$ and then through all $z_i^{(\ell)}$ that depend on $a_j^{(\ell-1)}$:

$$\frac{\partial L}{\partial z_j^{(\ell-1)}} = \frac{\partial L}{\partial a_j^{(\ell-1)}} \cdot \frac{\partial a_j^{(\ell-1)}}{\partial z_j^{(\ell-1)}}$$

First, compute $\frac{\partial L}{\partial a_j^{(\ell-1)}}$. The loss depends on $a_j^{(\ell-1)}$ through all $z_i^{(\ell)}$:

$$\frac{\partial L}{\partial a_j^{(\ell-1)}} = \sum_{i=1}^{n_\ell} \frac{\partial L}{\partial z_i^{(\ell)}} \cdot \frac{\partial z_i^{(\ell)}}{\partial a_j^{(\ell-1)}}$$

Since $z_i^{(\ell)} = \sum_{k} W_{ik}^{(\ell)} a_k^{(\ell-1)} + b_i^{(\ell)}$:

$$\frac{\partial z_i^{(\ell)}}{\partial a_j^{(\ell-1)}} = W_{ij}^{(\ell)}$$

Thus:

$$\frac{\partial L}{\partial a_j^{(\ell-1)}} = \sum_{i=1}^{n_\ell} \delta_i^{(\ell)} \cdot W_{ij}^{(\ell)}$$

In matrix form:

$$\frac{\partial L}{\partial \mathbf{a}^{(\ell-1)}} = (\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}$$

Finally:

$$\delta_j^{(\ell-1)} = \frac{\partial L}{\partial a_j^{(\ell-1)}} \cdot \sigma'^{(\ell-1)}(z_j^{(\ell-1)})$$

In vector form:

$$\boldsymbol{\delta}^{(\ell-1)} = [(\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}] \odot \sigma'^{(\ell-1)}(\mathbf{z}^{(\ell-1)})$$

This completes the derivation. The recursion for $\boldsymbol{\delta}^{(\ell)}$ allows us to compute gradients for all layers by propagating backward from the output.

### Conceptual Understanding

The formula $\boldsymbol{\delta}^{(\ell-1)} = [(\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}] \odot \sigma'^{(\ell-1)}(\mathbf{z}^{(\ell-1)})$ has a beautiful interpretation:

1. **$(\mathbf{W}^{(\ell)})^T \boldsymbol{\delta}^{(\ell)}$**: Errors from layer $\ell$ are propagated backward through the transpose of the weight matrix. The transpose captures how each neuron in layer $\ell-1$ contributes to each neuron in layer $\ell$.

2. **$\sigma'^{(\ell-1)}(\mathbf{z}^{(\ell-1)})$**: The propagated error is modulated by the local derivative of the activation function. If $\sigma'$ is small (e.g., in saturated regions of sigmoid), the gradient is attenuated—this is the vanishing gradient problem.

3. **Element-wise multiplication $\odot$**: Each neuron's error is scaled by its own activation derivative, independent of other neurons in the same layer.

## Computing Gradients for Common Activation Functions

The derivative $\sigma'(z)$ appears repeatedly in backpropagation. Let's compute it for common activations.

### ReLU

$$\sigma(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

Derivative:

$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

At $z = 0$, the derivative is undefined (the function has a kink). In practice, we define $\sigma'(0) = 0$ or $\sigma'(0) = 1$; the choice rarely matters.

**Key property**: For $z > 0$, $\sigma'(z) = 1$, so gradients pass through unchanged. This is why ReLU doesn't suffer from vanishing gradients for positive activations.

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Derivative: We'll derive this carefully.

Let $\sigma(z) = \frac{1}{1 + e^{-z}}$. Rewrite as $\sigma(z) = (1 + e^{-z})^{-1}$.

Using the chain rule:

$$\sigma'(z) = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

We can express this in terms of $\sigma$ itself:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}}$$

Note that $\frac{e^{-z}}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$.

Thus:

$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

This is a convenient form: we can compute the derivative from the forward pass output $\sigma(z)$ without needing $z$.

**Behavior**: 
- At $z = 0$: $\sigma(0) = 0.5$, so $\sigma'(0) = 0.5 \cdot 0.5 = 0.25$.
- As $z \to \infty$: $\sigma(z) \to 1$, so $\sigma'(z) \to 0$.
- As $z \to -\infty$: $\sigma(z) \to 0$, so $\sigma'(z) \to 0$.

The derivative vanishes for large $|z|$, causing vanishing gradients in deep networks.

### Tanh

$$\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Derivative: Using the quotient rule,

$$\sigma'(z) = \frac{(e^z + e^{-z})(e^z + e^{-z}) - (e^z - e^{-z})(e^z - e^{-z})}{(e^z + e^{-z})^2}$$

Simplify the numerator:

$$(e^z + e^{-z})^2 - (e^z - e^{-z})^2 = (e^{2z} + 2 + e^{-2z}) - (e^{2z} - 2 + e^{-2z}) = 4$$

Thus:

$$\sigma'(z) = \frac{4}{(e^z + e^{-z})^2}$$

We can express this as:

$$\sigma'(z) = 1 - \tanh^2(z) = 1 - \sigma(z)^2$$

**Behavior**: Similar to sigmoid, the derivative vanishes for large $|z|$.

### Leaky ReLU

$$\sigma(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

Derivative:

$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

The derivative is never zero (assuming $\alpha > 0$), preventing dying ReLU.

### Summary Table

| Activation $\sigma(z)$ | Derivative $\sigma'(z)$ |
|------------------------|-------------------------|
| ReLU: $\max(0, z)$ | $\mathbb{1}[z > 0]$ |
| Sigmoid: $\frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh: $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \sigma(z)^2$ |
| Leaky ReLU: $\max(\alpha z, z)$ | $\begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$ |

## Softmax and Cross-Entropy: Combined Gradient

For multi-class classification, we use softmax activation in the output layer and cross-entropy loss. Computing the gradient of this combination has a particularly elegant form.

### Setup

Let $\mathbf{z} \in \mathbb{R}^K$ be the logits (pre-softmax activations) at the output layer. The softmax function computes:

$$\hat{p}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

The cross-entropy loss for true class $y$ (where $y \in \{1, 2, \ldots, K\}$) is:

$$L = -\log \hat{p}_y = -\log \left( \frac{e^{z_y}}{\sum_{j=1}^{K} e^{z_j}} \right) = -z_y + \log \sum_{j=1}^{K} e^{z_j}$$

We want $\frac{\partial L}{\partial z_k}$ for $k = 1, 2, \ldots, K$.

### Derivation

$$\frac{\partial L}{\partial z_k} = \frac{\partial}{\partial z_k} \left[ -z_y + \log \sum_{j=1}^{K} e^{z_j} \right]$$

The first term contributes:

$$\frac{\partial (-z_y)}{\partial z_k} = \begin{cases} -1 & \text{if } k = y \\ 0 & \text{if } k \neq y \end{cases} = -\mathbb{1}[k = y]$$

For the second term, use the chain rule:

$$\frac{\partial}{\partial z_k} \log \sum_{j=1}^{K} e^{z_j} = \frac{1}{\sum_{j=1}^{K} e^{z_j}} \cdot \frac{\partial}{\partial z_k} \sum_{j=1}^{K} e^{z_j} = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = \hat{p}_k$$

Combining:

$$\frac{\partial L}{\partial z_k} = -\mathbb{1}[k = y] + \hat{p}_k = \hat{p}_k - \mathbb{1}[k = y]$$

### Interpretation

If we use one-hot encoding for the label, $\mathbf{y} = [0, \ldots, 1, \ldots, 0]^T$ where $y_k = 1$ if $k$ is the true class and $y_k = 0$ otherwise, then:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}$$

This is remarkably simple: the gradient is the difference between the predicted and true distributions. If the prediction is correct ($\hat{p}_y \approx 1$), the gradient is small. If the prediction is wrong ($\hat{p}_y \approx 0$), the gradient is large, providing strong learning signal.

### Computational Efficiency

When implementing softmax + cross-entropy, we don't compute them separately. Instead, we use the combined formula:

$$L = -z_y + \log \sum_{j=1}^{K} e^{z_j}$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}$$

This avoids numerical issues and is computationally efficient.

## Gradient Flow and Vanishing/Exploding Gradients

Understanding how gradients propagate through deep networks is crucial for successful training.

### Gradient Magnitude Analysis

Consider a deep network with $L$ layers. The gradient of the loss with respect to layer $\ell$ depends on the product of derivatives from all subsequent layers:

$$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} \propto \prod_{k=\ell+1}^{L} (\mathbf{W}^{(k)})^T \odot \sigma'^{(k)}(\mathbf{z}^{(k)})$$

(This is a simplification; the actual formula involves matrix products and element-wise operations, but the intuition holds.)

If each $\|\mathbf{W}^{(k)}\| \approx w$ and each $|\sigma'^{(k)}| \approx s$, then the gradient magnitude scales as:

$$\|\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}\| \propto (ws)^{L-\ell}$$

### Vanishing Gradients

If $ws < 1$ (small weights or small activation derivatives), then as $L - \ell$ grows (i.e., for early layers in a deep network), the gradient magnitude decreases exponentially:

$$(ws)^{L-\ell} \to 0 \text{ as } L - \ell \to \infty$$

This is the **vanishing gradient problem**. Early layers learn very slowly because their gradients are tiny.

**Causes**:
1. **Activation functions**: Sigmoid and tanh have $|\sigma'(z)| \leq 0.25$ and $|\sigma'(z)| \leq 1$ respectively. For large $|z|$, the derivatives approach zero.
2. **Small weights**: If weights are initialized too small, gradients shrink as they propagate backward.

**Solutions**:
1. **Use ReLU**: For $z > 0$, $\sigma'(z) = 1$, preventing gradient attenuation.
2. **Proper weight initialization**: Initialize weights with variance scaled appropriately (Xavier or He initialization).
3. **Batch normalization**: Normalizes activations, keeping them in a range where gradients are healthy.
4. **Residual connections**: Skip connections (as in ResNet) allow gradients to flow directly through the network.

### Exploding Gradients

If $ws > 1$, gradients grow exponentially:

$$(ws)^{L-\ell} \to \infty \text{ as } L - \ell \to \infty$$

This is the **exploding gradient problem**. Gradients become huge, causing numerical overflow and unstable training.

**Solutions**:
1. **Gradient clipping**: Cap gradient magnitude to a maximum value.
2. **Proper weight initialization**: Prevent large initial weights.
3. **Careful learning rate selection**: Use smaller learning rates if gradients are large.

### Gradient Norm Inspection

During training, it's useful to monitor gradient norms:

$$\|\nabla_{\mathbf{W}^{(\ell)}} L\| = \sqrt{\sum_{i,j} \left( \frac{\partial L}{\partial W_{ij}^{(\ell)}} \right)^2}$$

If gradients are consistently near zero (vanishing) or very large (exploding), training will fail. Tools like TensorBoard can visualize gradient histograms to diagnose these issues.

## Matrix Calculus and Vectorization

So far, we've derived gradients element-wise. For efficient implementation, we use matrix calculus and vectorized operations.

### Jacobian Matrices

For a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** is:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

For the affine transformation $\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$ where $\mathbf{W} \in \mathbb{R}^{m \times n}$, $\mathbf{a} \in \mathbb{R}^n$, the Jacobian with respect to $\mathbf{a}$ is:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{a}} = \mathbf{W}$$

For the element-wise activation $\mathbf{a} = \sigma(\mathbf{z})$, the Jacobian is diagonal:

$$\frac{\partial \mathbf{a}}{\partial \mathbf{z}} = \text{diag}(\sigma'(\mathbf{z}))$$

where $\text{diag}(\mathbf{v})$ is a diagonal matrix with $\mathbf{v}$ on the diagonal.

### Chain Rule in Matrix Form

For $\mathbf{y} = g(\mathbf{f}(\mathbf{x}))$, the chain rule gives:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{f}} \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$$

For scalar loss $L$ depending on vector $\mathbf{z}$:

$$\frac{\partial L}{\partial \mathbf{x}} = \left( \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right)^T \frac{\partial L}{\partial \mathbf{z}}$$

This is why we see transposes in backpropagation formulas.

### Gradient with Respect to Matrices

For a loss $L$ depending on matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$, the gradient is also an $m \times n$ matrix:

$$\frac{\partial L}{\partial \mathbf{W}} = \begin{bmatrix}
\frac{\partial L}{\partial W_{11}} & \frac{\partial L}{\partial W_{12}} & \cdots & \frac{\partial L}{\partial W_{1n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial W_{m1}} & \frac{\partial L}{\partial W_{m2}} & \cdots & \frac{\partial L}{\partial W_{mn}}
\end{bmatrix}$$

For the affine transformation $\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$, if we know $\frac{\partial L}{\partial \mathbf{z}} = \boldsymbol{\delta}$, then:

$$\frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta} \mathbf{a}^T$$

This is the outer product we derived earlier, but now we see it arises naturally from matrix calculus.

### Vectorization for Batches

For a batch of $B$ examples, we stack inputs into a matrix $\mathbf{X} \in \mathbb{R}^{B \times n_0}$. The forward pass becomes:

$$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)} (\mathbf{W}^{(\ell)})^T + \mathbf{1}_B (\mathbf{b}^{(\ell)})^T$$

where $\mathbf{1}_B$ is a column of ones and the bias is broadcast across the batch.

The backward pass accumulates gradients over the batch:

$$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} = \frac{1}{B} \sum_{i=1}^{B} \boldsymbol{\delta}^{(\ell)}_i (\mathbf{a}^{(\ell-1)}_i)^T = \frac{1}{B} (\boldsymbol{\Delta}^{(\ell)})^T \mathbf{A}^{(\ell-1)}$$

where $\boldsymbol{\Delta}^{(\ell)} \in \mathbb{R}^{B \times n_\ell}$ stacks all $\boldsymbol{\delta}^{(\ell)}_i$ row-wise.

This vectorization enables efficient computation on GPUs.

## Computational Complexity of Backpropagation

Backpropagation is remarkably efficient.

### Forward Pass Complexity

For a layer with weight matrix $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$:
- Matrix-vector multiplication $\mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)}$: $O(n_\ell \cdot n_{\ell-1})$
- Element-wise activation: $O(n_\ell)$

Total for the network: $O\left(\sum_{\ell=1}^{L} n_\ell \cdot n_{\ell-1}\right)$

For a batch of size $B$: $O\left(B \sum_{\ell=1}^{L} n_\ell \cdot n_{\ell-1}\right)$

### Backward Pass Complexity

For each layer:
- Computing $\boldsymbol{\delta}^{(\ell)}$: $O(n_\ell \cdot n_{\ell+1})$ (matrix-vector product)
- Computing $\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}$: $O(n_\ell \cdot n_{\ell-1})$ (outer product)

Total: Same order as forward pass, $O\left(\sum_{\ell=1}^{L} n_\ell \cdot n_{\ell-1}\right)$

**Key insight**: Backpropagation costs about the same as forward propagation. We get all gradients for roughly twice the cost of a single forward pass.

Compare this to naive finite differences, which would require $O(|\boldsymbol{\theta}|)$ forward passes (one per parameter), making it infeasible for large networks.

### Memory Complexity

During forward pass, we must **store all activations** $\mathbf{a}^{(\ell)}$ and pre-activations $\mathbf{z}^{(\ell)}$ because they're needed in the backward pass.

Memory: $O\left(\sum_{\ell=1}^{L} n_\ell\right)$ per example, or $O\left(B \sum_{\ell=1}^{L} n_\ell\right)$ for a batch.

For very deep networks (e.g., ResNets with 100+ layers), memory can be a bottleneck. Techniques like **gradient checkpointing** trade computation for memory by recomputing activations during the backward pass instead of storing them.

---

In this part, we've rigorously derived the backpropagation algorithm from first principles using the multivariate chain rule. We've shown how gradients flow backward through the network, computed derivatives for common activation functions, and analyzed the vanishing/exploding gradient problems. We've also covered the computational and memory complexity of training.

In the next part, we'll implement these derivations in code, building a neural network from scratch to solidify understanding, before examining how modern frameworks like TensorFlow/Keras handle these computations automatically.

## Detailed Derivations for Project-Specific Architectures

### ReLU Activation: Complete Analysis for Our Dense Layers

Our neuropathology classifier uses ReLU activation in both dense layers (lines 270 and 279 of `neuropathology_model.py`). Let's derive all properties rigorously.

**Definition**:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**First Derivative**:

For $z \neq 0$, the derivative is straightforward:
- For $z > 0$: $\frac{d}{dz}\text{ReLU}(z) = \frac{d}{dz}z = 1$
- For $z < 0$: $\frac{d}{dz}\text{ReLU}(z) = \frac{d}{dz}(0) = 0$

At $z = 0$, the function has a corner (non-differentiable). We define the **subdifferential** as $[0, 1]$, and in practice use either 0 or 1. Most implementations use 0:

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases} = \mathbb{1}[z > 0]$$

**Second Derivative**: 

$$\text{ReLU}''(z) = 0 \text{ for } z \neq 0$$

This means ReLU is piecewise linear with zero curvature except at the origin.

**Backpropagation Through ReLU** (used in our dense layers):

Consider layer $\ell$ with pre-activation $z_i^{(\ell)} = \sum_j W_{ij}^{(\ell)} a_j^{(\ell-1)} + b_i^{(\ell)}$ and activation $a_i^{(\ell)} = \text{ReLU}(z_i^{(\ell)})$.

Given $\frac{\partial L}{\partial a_i^{(\ell)}}$ from the subsequent layer, we compute:

$$\frac{\partial L}{\partial z_i^{(\ell)}} = \frac{\partial L}{\partial a_i^{(\ell)}} \cdot \frac{\partial a_i^{(\ell)}}{\partial z_i^{(\ell)}} = \frac{\partial L}{\partial a_i^{(\ell)}} \cdot \mathbb{1}[z_i^{(\ell)} > 0]$$

**Interpretation**: The gradient passes through unchanged if the neuron was active ($z > 0$) during the forward pass, and is blocked (set to zero) if the neuron was inactive ($z \leq 0$).

**Computational Efficiency**: This requires only:
1. One comparison per neuron
2. One multiplication per neuron

No exponentials, divisions, or other expensive operations.

**Statistical Properties**:

For zero-centered input distributions with moderate spread:
- Approximately 50% of neurons are active
- Sparse representations (remaining 50% output exactly 0)
- Gradient sparsity aids generalization

**Reference to Project Code**: In `neuropathology_model.py` line 270:
```python
layers.Dense(512, activation='relu', name='fc1')
```

This creates 512 neurons, each computing:
- Forward: $a_i = \max(0, \sum_j W_{ij} h_j + b_i)$ where $h$ is the 1280-dim input from Global Average Pooling
- Backward: $\delta_i = \delta_{\text{next}} \cdot \mathbb{1}[z_i > 0]$ where $\delta_{\text{next}}$ comes from dropout

**Visualization**: See `../visualizations/activation_functions_comprehensive.png` for plots of ReLU and its derivative.

### Softmax and Cross-Entropy: The Output Layer

Our classifier's output layer (line 287) uses softmax with 17 outputs. Let's derive the complete mathematics.

**Softmax Function**:

Given logits $\mathbf{z} = [z_1, z_2, \ldots, z_K]^T$ where $K=17$ for our classifier:

$$\hat{p}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = \frac{e^{z_k}}{Z}$$

where $Z = \sum_{j=1}^{K} e^{z_j}$ is the partition function.

**Detailed Jacobian Derivation**:

The softmax Jacobian is crucial for backpropagation. We need $\frac{\partial \hat{p}_i}{\partial z_j}$ for all $i, j$.

**Case 1: $i = j$ (diagonal elements)**

$$\frac{\partial \hat{p}_i}{\partial z_i} = \frac{\partial}{\partial z_i} \left( \frac{e^{z_i}}{Z} \right)$$

Using the quotient rule:

$$= \frac{e^{z_i} \cdot Z - e^{z_i} \cdot \frac{\partial Z}{\partial z_i}}{Z^2}$$

Since $\frac{\partial Z}{\partial z_i} = e^{z_i}$:

$$= \frac{e^{z_i} \cdot Z - e^{z_i} \cdot e^{z_i}}{Z^2} = \frac{e^{z_i}}{Z} \cdot \frac{Z - e^{z_i}}{Z} = \hat{p}_i (1 - \hat{p}_i)$$

**Case 2: $i \neq j$ (off-diagonal elements)**

$$\frac{\partial \hat{p}_i}{\partial z_j} = \frac{\partial}{\partial z_j} \left( \frac{e^{z_i}}{Z} \right)$$

The numerator $e^{z_i}$ doesn't depend on $z_j$, so:

$$= \frac{0 \cdot Z - e^{z_i} \cdot \frac{\partial Z}{\partial z_j}}{Z^2} = \frac{-e^{z_i} \cdot e^{z_j}}{Z^2} = -\frac{e^{z_i}}{Z} \cdot \frac{e^{z_j}}{Z} = -\hat{p}_i \hat{p}_j$$

**Combined Result** using Kronecker delta $\delta_{ij}$:

$$\frac{\partial \hat{p}_i}{\partial z_j} = \hat{p}_i (\delta_{ij} - \hat{p}_j)$$

**Verification**:
- When $i = j$: $\hat{p}_i(\delta_{ii} - \hat{p}_i) = \hat{p}_i(1 - \hat{p}_i)$ ✓
- When $i \neq j$: $\hat{p}_i(\delta_{ij} - \hat{p}_j) = \hat{p}_i(0 - \hat{p}_j) = -\hat{p}_i \hat{p}_j$ ✓

**Cross-Entropy Loss**:

For true class $y \in \{1, 2, \ldots, K\}$:

$$L = -\log \hat{p}_y$$

**Combined Gradient** (the elegant result):

We want $\frac{\partial L}{\partial z_k}$ for all $k$:

$$\frac{\partial L}{\partial z_k} = \sum_{i=1}^{K} \frac{\partial L}{\partial \hat{p}_i} \cdot \frac{\partial \hat{p}_i}{\partial z_k}$$

First, $\frac{\partial L}{\partial \hat{p}_i}$:

$$L = -\log \hat{p}_y \implies \frac{\partial L}{\partial \hat{p}_i} = \begin{cases} -\frac{1}{\hat{p}_y} & \text{if } i = y \\ 0 & \text{if } i \neq y \end{cases}$$

Therefore:

$$\frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial \hat{p}_y} \cdot \frac{\partial \hat{p}_y}{\partial z_k} = -\frac{1}{\hat{p}_y} \cdot \hat{p}_y(\delta_{yk} - \hat{p}_k) = -(\delta_{yk} - \hat{p}_k) = \hat{p}_k - \delta_{yk}$$

where $\delta_{yk} = 1$ if $k = y$ and 0 otherwise.

**Using one-hot encoding** $\mathbf{y} \in \{0,1\}^K$ where $y_k = 1$ if $k$ is the true class:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}$$

**This is remarkably simple**: The gradient is just the prediction error!

**Numerical Example** (for our 17-class problem):

Suppose the true class is Meningioma T1C+ (class 5), and our network outputs:

$$\mathbf{z} = [2.1, 1.3, 0.8, 1.5, 5.2, 3.1, 0.9, \ldots] \in \mathbb{R}^{17}$$

After softmax:
$$\hat{\mathbf{p}} = [0.03, 0.01, 0.01, 0.02, 0.68, 0.08, 0.01, \ldots]$$

The one-hot label is:
$$\mathbf{y} = [0, 0, 0, 0, 1, 0, 0, \ldots]$$

The gradient is:
$$\frac{\partial L}{\partial \mathbf{z}} = [0.03, 0.01, 0.01, 0.02, -0.32, 0.08, 0.01, \ldots]$$

**Interpretation**:
- Class 5 (true class): gradient is $-0.32$, pushing logit $z_5$ **higher**
- All other classes: positive gradients, pushing their logits **lower**
- Magnitude proportional to predicted probability (higher confidence = larger gradient)

**Reference to Project Code**: In `neuropathology_model.py`:
- Line 287: `layers.Dense(self.num_classes, activation='softmax', name='output')`
- Line 400: `loss='categorical_crossentropy'`

TensorFlow/Keras automatically combines these for numerically stable computation.

**Visualization**: See `../visualizations/softmax_properties.png` and `../visualizations/cross_entropy_loss.png`.

## Comprehensive Gradient Derivations for All Common Activations

### Sigmoid: Historical Perspective and Modern Usage

**Function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**First Derivative** (detailed proof):

Let $u = 1 + e^{-z}$, so $\sigma(z) = u^{-1}$.

$$\frac{d\sigma}{dz} = \frac{d}{dz}(u^{-1}) = -u^{-2} \cdot \frac{du}{dz}$$

Since $\frac{du}{dz} = -e^{-z}$:

$$\frac{d\sigma}{dz} = -\frac{1}{(1 + e^{-z})^2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

**Expressing in terms of $\sigma(z)$**:

$$\frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$$

Note that:
$$\frac{e^{-z}}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$$

Therefore:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Second Derivative**:

$$\sigma''(z) = \frac{d}{dz}[\sigma(z)(1 - \sigma(z))]$$

Using the product rule:
$$= \sigma'(z)(1 - \sigma(z)) + \sigma(z) \cdot (-\sigma'(z))$$= \sigma'(z)(1 - 2\sigma(z))$$= \sigma(z)(1 - \sigma(z))(1 - 2\sigma(z))$$

**Critical Points**:
- $\sigma''(z) = 0$ when $\sigma(z) = 0, 1,$ or $0.5$
- Inflection point at $z = 0$ where $\sigma(0) = 0.5$

**Vanishing Gradient Analysis**:

The maximum derivative occurs at $z = 0$:
$$\sigma'(0) = \sigma(0)(1 - \sigma(0)) = 0.5 \times 0.5 = 0.25$$

For deep networks, gradients are multiplied across layers. With $L$ layers:
$$\left|\frac{\partial L}{\partial z^{(1)}}\right| \leq 0.25^L \cdot \text{const}$$

For $L = 10$: $0.25^{10} \approx 9.5 \times 10^{-7}$ (gradient essentially vanishes!)

**Why Not Used in Hidden Layers**: This rapid gradient decay makes sigmoid impractical for deep networks.

**Visualization**: See first panel of `../visualizations/activation_functions_comprehensive.png`.

### Hyperbolic Tangent (Tanh)

**Function**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}$$

**Relation to Sigmoid**:
$$\tanh(z) = 2\sigma(2z) - 1$$

**First Derivative** (using quotient rule):

Let $u = e^z - e^{-z}$ and $v = e^z + e^{-z}$.

$$\frac{du}{dz} = e^z + e^{-z} = v, \quad \frac{dv}{dz} = e^z - e^{-z} = u$$

By quotient rule:
$$\tanh'(z) = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2}$$

Computing $v^2 - u^2$:
$$v^2 - u^2 = (e^z + e^{-z})^2 - (e^z - e^{-z})^2$$= (e^{2z} + 2 + e^{-2z}) - (e^{2z} - 2 + e^{-2z}) = 4$$

Therefore:
$$\tanh'(z) = \frac{4}{(e^z + e^{-z})^2} = 1 - \tanh^2(z)$$

**Verification** at $z = 0$:
$$\tanh(0) = 0 \implies \tanh'(0) = 1 - 0^2 = 1$$ ✓

**Second Derivative**:
$$\tanh''(z) = -2\tanh(z) \cdot \tanh'(z) = -2\tanh(z)(1 - \tanh^2(z))$$

**Advantages over Sigmoid**:
1. **Zero-centered**: Outputs range from $-1$ to $1$, mean around 0
2. **Stronger gradient**: Maximum derivative is 1 (vs. 0.25 for sigmoid)
3. **Symmetric**: $\tanh(-z) = -\tanh(z)$

**Still suffers from vanishing gradients**, but less severely than sigmoid.

**Visualization**: See `../visualizations/activation_functions_comprehensive.png`, third panel.

### Leaky ReLU and Parametric ReLU (PReLU)

**Leaky ReLU**:
$$f(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases} = \max(\alpha z, z)$$

where $\alpha$ is a small positive constant (typically $\alpha = 0.01$).

**Derivative**:
$$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

**Key Advantage**: Non-zero gradient for negative inputs prevents "dying ReLU" problem.

**Parametric ReLU (PReLU)**: Treats $\alpha$ as a **learnable parameter**:

During training, we also compute:
$$\frac{\partial L}{\partial \alpha} = \sum_{i : z_i < 0} \frac{\partial L}{\partial f(z_i)} \cdot z_i$$

**Why it helps**: The network learns the optimal slope for negative values, adapting to the data.

**Visualization**: See `../visualizations/activation_functions_comprehensive.png`, fourth panel.

## Advanced Mathematical Tools for Deep Learning

### Matrix Calculus: Vector-by-Vector Derivatives

When dealing with vector-valued functions, we need systematic rules for derivatives.

**Setup**: Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ be a vector-valued function.

The **Jacobian matrix** is:
$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Chain Rule for Vectors**:

If $\mathbf{y} = \mathbf{f}(\mathbf{x})$ and $\mathbf{z} = \mathbf{g}(\mathbf{y})$, then:
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

**Example: Affine Transformation**

Let $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ where $\mathbf{W} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

Component-wise: $z_i = \sum_{j=1}^n W_{ij} x_j + b_i$

Jacobian:
$$\frac{\partial z_i}{\partial x_j} = W_{ij} \implies \frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}$$

**Backpropagation through Affine Layer**:

Given $\frac{\partial L}{\partial \mathbf{z}}$ (a column vector), we need $\frac{\partial L}{\partial \mathbf{x}}$:

$$\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right)^T \frac{\partial L}{\partial \mathbf{z}} = \mathbf{W}^T \frac{\partial L}{\partial \mathbf{z}}$$

This is why we see $\mathbf{W}^T$ in backpropagation!

**Gradient with Respect to Weight Matrix**:

For $\mathbf{z} = \mathbf{W}\mathbf{x}$, we want $\frac{\partial L}{\partial \mathbf{W}}$.

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot x_j$$

In matrix form:
$$\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^T$$

This is the **outer product** formula used in backpropagation.

### Information Theory Perspective on Cross-Entropy

**Entropy**: Measures uncertainty in a distribution:
$$H(P) = -\sum_{i=1}^K P(i) \log P(i)$$

**Cross-Entropy** between distributions $P$ and $Q$:
$$H(P, Q) = -\sum_{i=1}^K P(i) \log Q(i)$$

**KL Divergence** (Kullback-Leibler):
$$D_{KL}(P \| Q) = \sum_{i=1}^K P(i) \log \frac{P(i)}{Q(i)} = H(P, Q) - H(P)$$

**Minimizing Cross-Entropy = Minimizing KL Divergence**:

Since $H(P)$ is constant (true distribution doesn't change), minimizing $H(P, Q)$ is equivalent to minimizing $D_{KL}(P \| Q)$.

**For Classification**:
- $P$ is the true distribution (one-hot: all mass on correct class)
- $Q$ is the predicted distribution (softmax output)
- Minimizing cross-entropy makes predictions match the true distribution

**Maximum Likelihood Connection**:

The cross-entropy loss:
$$L = -\log Q(y) = -\log \hat{p}_y$$

is the negative log-likelihood of the true class under the model's distribution.

Minimizing $L$ ⟺ Maximizing likelihood ⟺ Maximum Likelihood Estimation (MLE)

## Numerical Stability and Implementation Considerations

### Numerically Stable Softmax

**Problem**: Computing $e^{z_k}$ can overflow for large $z_k$ (e.g., $e^{1000}$ = infinity in floating point).

**Solution**: Use the translation invariance property:
$$\text{softmax}(\mathbf{z})_k = \text{softmax}(\mathbf{z} - c)_k \quad \text{for any constant } c$$

**Proof**:
$$\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_k} e^{-c}}{\sum_j e^{z_j} e^{-c}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

**Best choice**: $c = \max_j z_j$, ensuring all exponents are $\leq 0$.

**Implementation**:
```python
def softmax_stable(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

**Reference**: TensorFlow/Keras automatically uses this stable version internally.

### Log-Sum-Exp Trick

When computing cross-entropy, we often need:
$$\log \sum_{j=1}^K e^{z_j}$$

**Stable computation**:
$$\log \sum_{j=1}^K e^{z_j} = c + \log \sum_{j=1}^K e^{z_j - c}$$

where $c = \max_j z_j$.

This appears in the cross-entropy formula:
$$L = -z_y + \log \sum_{j=1}^K e^{z_j}$$

### Gradient Checking: Verifying Backpropagation

**Finite Differences** provide a numerical approximation to gradients:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

for small $\epsilon$ (typically $10^{-4}$).

**Procedure**:
1. Implement backpropagation analytically
2. Compute numerical gradient using finite differences
3. Compare: $\left|\frac{\partial L}{\partial \theta}_{\text{analytical}} - \frac{\partial L}{\partial \theta}_{\text{numerical}}\right|$ should be $< 10^{-5}$

**Why This Works**: Taylor expansion:
$$L(\theta + \epsilon) = L(\theta) + \epsilon \frac{\partial L}{\partial \theta} + O(\epsilon^2)$$L(\theta - \epsilon) = L(\theta) - \epsilon \frac{\partial L}{\partial \theta} + O(\epsilon^2)$$

Subtracting:
$$L(\theta + \epsilon) - L(\theta - \epsilon) = 2\epsilon \frac{\partial L}{\partial \theta} + O(\epsilon^3)$$

The $O(\epsilon^2)$ terms cancel, giving better accuracy than one-sided difference.

**Reference to Code**: See Part D for implementation of gradient checking in NumPy.

## Theoretical Guarantees and Convergence Analysis

### Convergence of Gradient Descent (Simplified)

For a **convex** function $L(\theta)$ with **Lipschitz continuous gradient** (i.e., $\|\nabla L(\theta_1) - \nabla L(\theta_2)\| \leq C \|\theta_1 - \theta_2\|$):

**Theorem**: Gradient descent with step size $\eta < \frac{2}{C}$ converges to the global minimum.

**Proof Sketch**:
1. By Lipschitz continuity, $L(\theta - \eta \nabla L(\theta)) \leq L(\theta) - \eta \|\nabla L(\theta)\|^2 + \frac{C\eta^2}{2} \|\nabla L(\theta)\|^2$
2. Choosing $\eta < \frac{2}{C}$ ensures the second-order term is dominated
3. Thus $L$ decreases at each iteration
4. For convex $L$, this implies convergence to global minimum

**Reality for Neural Networks**:
- Loss is **not convex**
- Convergence to global minimum **not guaranteed**
- But local minima are often good enough in practice

### Why Deep Networks Don't Get Stuck in Bad Local Minima

**Empirical Observation**: For overparameterized networks, most local minima have similar loss values close to the global minimum.

**Theoretical Insight** (Choromanska et al., 2015): For certain random network models, the loss surface resembles a high-dimensional landscape where:
- Number of local minima grows exponentially with network size
- But most local minima have low loss
- Saddle points dominate over poor local minima

**Escape from Saddle Points**: SGD with noise can escape saddle points efficiently. At a saddle point:
- Gradient is zero: $\nabla L = 0$
- Hessian has negative eigenvalues
- Adding noise (from minibatches) pushes optimizer away from saddle

## Visualization and Geometric Interpretation

### The Error Surface

For a simple 2-parameter network, we can visualize the loss surface $L(\theta_1, \theta_2)$.

**Characteristics**:
- **Ravines**: Long, narrow valleys with steep sides
- **Saddle points**: Gradients zero but not minima
- **Plateaus**: Flat regions with small gradients

**See**: `../visualizations/` directory (2D loss contours would be generated for specific examples).

### Gradient Flow Through Layers

The magnitude of gradients typically varies across layers. In a well-trained network:
- **Early layers**: Smaller gradients (further from loss)
- **Later layers**: Larger gradients (closer to loss)

**See**: `../visualizations/gradient_flow.png` illustrating vanishing and exploding gradients.

### The Chain Rule as a Computational Graph

**Representation**: Each operation in the network is a node, data flows along edges.

**Forward pass**: Data flows from input to output
**Backward pass**: Gradients flow from output to input

**See**: `../visualizations/chain_rule_visualization.png` for a concrete example.

---

This expansion provides significantly more mathematical depth, covering:
- Detailed derivations for all activation functions used in the project
- Complete softmax and cross-entropy mathematics
- Matrix calculus foundations
- Information theory perspective
- Numerical stability considerations
- Theoretical convergence guarantees
- Visualization references

All derivations are step-by-step with complete proofs, and directly reference the project code and generated visualizations.
