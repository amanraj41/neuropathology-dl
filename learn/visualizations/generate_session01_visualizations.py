"""
Generate visualizations for Session 01: Neural Networks

This script generates all mathematical visualizations referenced in the tutorial.
Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_plot(filename):
    """Save plot with consistent settings."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def visualize_activation_functions():
    """Generate activation function plots with derivatives."""
    z = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Activation Functions and Their Derivatives', fontsize=16, fontweight='bold')
    
    # ReLU
    relu = np.maximum(0, z)
    relu_grad = (z > 0).astype(float)
    axes[0, 0].plot(z, relu, 'b-', linewidth=2, label='ReLU(z)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('ReLU Activation', fontweight='bold')
    axes[0, 0].set_xlabel('z')
    axes[0, 0].set_ylabel('σ(z)')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].legend()
    
    axes[1, 0].plot(z, relu_grad, 'r-', linewidth=2, label="ReLU'(z)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('ReLU Derivative', fontweight='bold')
    axes[1, 0].set_xlabel('z')
    axes[1, 0].set_ylabel("σ'(z)")
    axes[1, 0].set_ylim([-0.2, 1.2])
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].legend()
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    axes[0, 1].plot(z, sigmoid, 'b-', linewidth=2, label='σ(z) = 1/(1+e⁻ᶻ)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Sigmoid Activation', fontweight='bold')
    axes[0, 1].set_xlabel('z')
    axes[0, 1].set_ylabel('σ(z)')
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='0.5')
    axes[0, 1].axhline(y=1, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 1].plot(z, sigmoid_grad, 'r-', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Sigmoid Derivative', fontweight='bold')
    axes[1, 1].set_xlabel('z')
    axes[1, 1].set_ylabel("σ'(z)")
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0.25, color='g', linestyle='--', alpha=0.5, label='max = 0.25')
    axes[1, 1].legend()
    
    # Tanh
    tanh = np.tanh(z)
    tanh_grad = 1 - tanh**2
    axes[0, 2].plot(z, tanh, 'b-', linewidth=2, label='tanh(z)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_title('Tanh Activation', fontweight='bold')
    axes[0, 2].set_xlabel('z')
    axes[0, 2].set_ylabel('σ(z)')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=1, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=-1, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].legend()
    
    axes[1, 2].plot(z, tanh_grad, 'r-', linewidth=2, label="tanh'(z) = 1 - tanh²(z)")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_title('Tanh Derivative', fontweight='bold')
    axes[1, 2].set_xlabel('z')
    axes[1, 2].set_ylabel("σ'(z)")
    axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 2].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='max = 1')
    axes[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 2].legend()
    
    # Leaky ReLU
    alpha = 0.01
    leaky_relu = np.where(z > 0, z, alpha * z)
    leaky_relu_grad = np.where(z > 0, 1, alpha)
    axes[0, 3].plot(z, leaky_relu, 'b-', linewidth=2, label=f'Leaky ReLU (α={alpha})')
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].set_title('Leaky ReLU Activation', fontweight='bold')
    axes[0, 3].set_xlabel('z')
    axes[0, 3].set_ylabel('σ(z)')
    axes[0, 3].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 3].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 3].legend()
    
    axes[1, 3].plot(z, leaky_relu_grad, 'r-', linewidth=2, label=f"Leaky ReLU' (α={alpha})")
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_title('Leaky ReLU Derivative', fontweight='bold')
    axes[1, 3].set_xlabel('z')
    axes[1, 3].set_ylabel("σ'(z)")
    axes[1, 3].set_ylim([-0.05, 1.2])
    axes[1, 3].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 3].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='1')
    axes[1, 3].axhline(y=alpha, color='orange', linestyle='--', alpha=0.5, label=f'α={alpha}')
    axes[1, 3].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 3].legend()
    
    plt.tight_layout()
    save_plot('activation_functions_comprehensive.png')


def visualize_softmax():
    """Visualize softmax function behavior."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Softmax Function Properties', fontsize=14, fontweight='bold')
    
    # Case 1: Equal logits
    z1 = np.array([1, 1, 1, 1, 1])
    p1 = np.exp(z1) / np.sum(np.exp(z1))
    axes[0].bar(range(len(p1)), p1, color='steelblue', edgecolor='black')
    axes[0].set_title('Equal Logits\nz = [1, 1, 1, 1, 1]', fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Probability')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='1/K = 0.2')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    # Case 2: One dominant logit
    z2 = np.array([10, 1, 1, 1, 1])
    p2 = np.exp(z2) / np.sum(np.exp(z2))
    axes[1].bar(range(len(p2)), p2, color='steelblue', edgecolor='black')
    axes[1].set_title('Dominant Logit\nz = [10, 1, 1, 1, 1]', fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(0, p2[0] + 0.05, f'{p2[0]:.4f}', ha='center', fontweight='bold')
    
    # Case 3: Temperature effect
    z3 = np.array([5, 3, 2, 1, 0.5])
    temperatures = [0.5, 1.0, 2.0]
    colors = ['darkblue', 'steelblue', 'lightblue']
    x = np.arange(len(z3))
    width = 0.25
    
    for i, (temp, color) in enumerate(zip(temperatures, colors)):
        p3 = np.exp(z3 / temp) / np.sum(np.exp(z3 / temp))
        axes[2].bar(x + i*width, p3, width, label=f'T={temp}', color=color, edgecolor='black')
    
    axes[2].set_title('Temperature Scaling\nz = [5, 3, 2, 1, 0.5]', fontweight='bold')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Probability')
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(range(len(z3)))
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot('softmax_properties.png')


def visualize_gradient_flow():
    """Visualize vanishing and exploding gradients."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Gradient Flow Through Deep Networks', fontsize=14, fontweight='bold')
    
    # Vanishing gradients
    layers = np.arange(1, 21)
    
    # Sigmoid network (gradient shrinks)
    sigmoid_grad_sigmoid = 0.25  # max derivative of sigmoid
    weight_scale = 0.5
    vanishing_grad = (sigmoid_grad_sigmoid * weight_scale) ** layers
    
    axes[0].semilogy(layers, vanishing_grad, 'r-o', linewidth=2, markersize=6, label='Sigmoid (vanishing)')
    
    # ReLU network (gradient stable)
    relu_grad = 1.0
    weight_scale_relu = 1.0
    stable_grad = np.ones_like(layers)
    axes[0].semilogy(layers, stable_grad, 'g-s', linewidth=2, markersize=6, label='ReLU (stable)')
    
    axes[0].set_xlabel('Layer Depth', fontsize=12)
    axes[0].set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    axes[0].set_title('Vanishing Gradients', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=1e-8, color='orange', linestyle='--', alpha=0.7, label='Numerical precision limit')
    
    # Exploding gradients
    # Poor initialization (large weights)
    weight_scales = [1.1, 1.2, 1.5]
    colors = ['orange', 'red', 'darkred']
    
    for ws, color in zip(weight_scales, colors):
        exploding_grad = ws ** layers
        # Clip at 1e10 for visualization
        exploding_grad = np.minimum(exploding_grad, 1e10)
        axes[1].semilogy(layers, exploding_grad, marker='o', color=color, linewidth=2, 
                        markersize=6, label=f'Weight scale = {ws}')
    
    # Proper initialization
    axes[1].semilogy(layers, stable_grad, 'g-s', linewidth=2, markersize=6, label='Proper init (stable)')
    
    axes[1].set_xlabel('Layer Depth', fontsize=12)
    axes[1].set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    axes[1].set_title('Exploding Gradients', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].axhline(y=1e10, color='purple', linestyle='--', alpha=0.7, label='Overflow threshold')
    
    plt.tight_layout()
    save_plot('gradient_flow.png')


def visualize_chain_rule():
    """Visualize chain rule in action."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('The Chain Rule: Computational Graph and Gradient Flow', fontsize=14, fontweight='bold')
    
    # Example: y = (3x + 5)^2
    x_vals = np.linspace(-3, 3, 100)
    u_vals = 3 * x_vals + 5
    y_vals = u_vals ** 2
    
    # Forward pass
    axes[0, 0].plot(x_vals, u_vals, 'b-', linewidth=2, label='u = 3x + 5')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('u', fontsize=11)
    axes[0, 0].set_title('Forward: x → u', fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    
    axes[0, 1].plot(u_vals, y_vals, 'r-', linewidth=2, label='y = u²')
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('u', fontsize=11)
    axes[0, 1].set_ylabel('y', fontsize=11)
    axes[0, 1].set_title('Forward: u → y', fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    
    # Derivatives (backward pass)
    du_dx = 3 * np.ones_like(x_vals)
    dy_du = 2 * u_vals
    dy_dx = dy_du * du_dx
    
    axes[1, 0].plot(x_vals, du_dx, 'b-', linewidth=2, label='∂u/∂x = 3')
    axes[1, 0].axhline(y=3, color='b', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('Derivative', fontsize=11)
    axes[1, 0].set_title('Backward: ∂u/∂x (constant)', fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_ylim([0, 5])
    
    axes[1, 1].plot(x_vals, dy_dx, 'g-', linewidth=3, label='∂y/∂x = (∂y/∂u)(∂u/∂x)')
    axes[1, 1].plot(x_vals, dy_du, 'r--', linewidth=2, alpha=0.7, label='∂y/∂u = 2u')
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('Derivative', fontsize=11)
    axes[1, 1].set_title('Chain Rule: ∂y/∂x', fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    save_plot('chain_rule_visualization.png')


def visualize_cross_entropy():
    """Visualize cross-entropy loss behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Entropy Loss Function', fontsize=14, fontweight='bold')
    
    # Loss as a function of predicted probability
    p_pred = np.linspace(0.001, 0.999, 1000)
    loss_correct = -np.log(p_pred)  # When prediction matches true class
    loss_wrong = -np.log(1 - p_pred)  # When prediction is for wrong class
    
    axes[0].plot(p_pred, loss_correct, 'g-', linewidth=2.5, label='L = -log(p) [correct class]')
    axes[0].plot(p_pred, loss_wrong, 'r-', linewidth=2.5, label='L = -log(1-p) [wrong class]')
    axes[0].set_xlabel('Predicted Probability p', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Cross-Entropy vs. Predicted Probability', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([0, 5])
    axes[0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='p=0.5')
    axes[0].text(0.9, 4, 'Low loss\n(confident & correct)', fontsize=10, ha='center', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[0].text(0.1, 4, 'High loss\n(confident & wrong)', fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Gradient of cross-entropy + softmax
    # For true class k: ∂L/∂z_k = p_k - 1
    # For other classes: ∂L/∂z_i = p_i
    
    p_true_class = np.linspace(0.001, 0.999, 1000)
    grad_true_class = p_true_class - 1  # ∂L/∂z for correct class
    grad_other_class = p_true_class  # ∂L/∂z for incorrect class
    
    axes[1].plot(p_true_class, grad_true_class, 'g-', linewidth=2.5, label='∂L/∂z [correct class]')
    axes[1].plot(p_true_class, grad_other_class, 'r-', linewidth=2.5, label='∂L/∂z [wrong class]')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1].axvline(x=1.0, color='g', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Predicted Probability p', fontsize=12)
    axes[1].set_ylabel('Gradient ∂L/∂z', fontsize=12)
    axes[1].set_title('Gradient of Softmax + Cross-Entropy', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].text(0.5, -0.5, 'Gradient → 0\nas p → 1', fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[1].text(0.5, 0.5, 'Large gradient\nwhen wrong', fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    save_plot('cross_entropy_loss.png')


def main():
    """Generate all visualizations."""
    print("Generating Session 01 visualizations...")
    print("=" * 60)
    
    visualize_activation_functions()
    visualize_softmax()
    visualize_gradient_flow()
    visualize_chain_rule()
    visualize_cross_entropy()
    
    print("=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
