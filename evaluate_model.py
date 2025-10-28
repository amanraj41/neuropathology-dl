"""
Model Evaluation Script for Neuropathology Detection

This script evaluates trained Keras models on a test dataset and generates
comprehensive metrics JSON files that can be used by app.py to display
model performance metrics.

Usage:
    python evaluate_model.py --model_path models/best_model.keras --data_dir /path/to/test/data
    python evaluate_model.py --model_path models/final_model.keras --data_dir /path/to/test/data --output models/final_model_metrics.json
"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel
from src.data.data_loader import DataGenerator
from src.utils.helpers import ModelEvaluator, Visualizer, get_class_names


def load_dataset(data_dir: str):
    """
    Load dataset from directory structure.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Tuple of (image_paths, labels, class_names)
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    image_paths = []
    labels = []
    class_names = []
    
    # Get class directories
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"\nFound {len(class_dirs)} classes:")
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        class_images = [os.path.join(class_path, f) 
                       for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  {class_name}: {len(class_images)} images")
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
        class_names.append(class_name)
    
    # Convert to numpy arrays
    labels = np.array(labels)
    
    # One-hot encode labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    
    return image_paths, labels_onehot, class_names


def evaluate_model(model_path: str, data_dir: str, output_path: str = None, batch_size: int = 32):
    """
    Evaluate a trained model and generate metrics JSON.
    
    Args:
        model_path: Path to trained model (.keras file)
        data_dir: Path to test data directory
        output_path: Path to save metrics JSON (default: same name as model with _metrics.json suffix)
        batch_size: Batch size for evaluation
    """
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model_wrapper = NeuropathologyModel()
    model_wrapper.load_model(model_path)
    model = model_wrapper.model
    
    # Load dataset
    print(f"\nLoading test dataset from: {data_dir}")
    image_paths, labels, class_names = load_dataset(data_dir)
    
    # Use only test split (15% of data)
    _, test_paths, _, test_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, 
        stratify=np.argmax(labels, axis=1)
    )
    
    print(f"\nTest set size: {len(test_paths)} images")
    
    # Create test generator
    test_gen = DataGenerator(
        test_paths, test_labels,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate_model(model, test_gen, class_names)
    
    # Print evaluation report
    evaluator.print_evaluation_report(evaluation)
    
    # Prepare metrics for JSON output
    metrics = {
        'model_path': model_path,
        'test_samples': len(test_paths),
        'num_classes': len(class_names),
        'accuracy': float(evaluation['accuracy']),
        'classification_report': evaluation['classification_report'],
        'class_names': class_names
    }
    
    # Determine output path
    if output_path is None:
        # Default: same directory as model with _metrics.json suffix
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}_metrics.json"
    
    # Save metrics to JSON
    print(f"\nSaving metrics to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved successfully")
    
    # Generate visualizations
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    
    # Confusion matrix
    cm_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
    visualizer.plot_confusion_matrix(
        evaluation['y_true'],
        evaluation['y_pred'],
        class_names,
        save_path=cm_path
    )
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Metrics File: {output_path}")


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained neuropathology detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Evaluate best_model.keras on test data
    python evaluate_model.py --model_path models/best_model.keras --data_dir ./data/brain_mri_17
    
    # Evaluate final_model.keras with custom output path
    python evaluate_model.py --model_path models/final_model.keras --data_dir ./data/brain_mri_17 --output models/final_model_metrics.json
    
    # Evaluate finetuned model
    python evaluate_model.py --model_path models/best_model_finetuned.keras --data_dir ./data/brain_mri_17
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.keras file)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save metrics JSON (default: model_name_metrics.json)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        parser.error(f"Model file does not exist: {args.model_path}")
    
    if not os.path.exists(args.data_dir):
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    # Run evaluation
    evaluate_model(args.model_path, args.data_dir, args.output, args.batch_size)


if __name__ == "__main__":
    main()
