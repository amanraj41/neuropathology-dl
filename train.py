"""
Training Script for Neuropathology Detection Model

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model creation and compilation
- Two-stage training (feature extraction + fine-tuning)
- Model evaluation and saving

Usage:
    python train.py --data_dir /path/to/data --epochs 30 --batch_size 32

Requirements:
    - Organized dataset with subdirectories for each class
    - Sufficient GPU memory (recommended: 8GB+)
"""

import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel, create_callbacks
from src.data.data_loader import DataGenerator, MRIDataLoader
from src.utils.helpers import Visualizer, ModelEvaluator, get_class_names


def load_dataset(data_dir: str, img_size=(224, 224)):
    """
    Load dataset from directory structure.
    
    Expected structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
        ...
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size
    
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


def create_data_generators(image_paths, labels, batch_size=32, 
                          validation_split=0.15, test_split=0.15):
    """
    Create train, validation, and test data generators.
    
    Args:
        image_paths: List of image paths
        labels: One-hot encoded labels
        batch_size: Batch size for training
        validation_split: Proportion of data for validation
        test_split: Proportion of data for testing
    
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    # Prepare class indices for stratification (handles one-hot labels)
    if isinstance(labels, np.ndarray) and labels.ndim == 2:
        labels_cls = np.argmax(labels, axis=1)
    else:
        labels_cls = labels

    # Split data
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels, train_val_cls, test_cls = train_test_split(
        image_paths, labels, labels_cls, test_size=test_split, random_state=42, stratify=labels_cls
    )
    
    # Second split: separate validation set
    val_size = validation_split / (1 - test_split)
    train_paths, val_paths, train_labels, val_labels, train_cls, val_cls = train_test_split(
        train_val_paths, train_val_labels, train_val_cls, test_size=val_size, 
        random_state=42, stratify=train_val_cls
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Testing:    {len(test_paths)} images")
    
    # Create generators
    train_gen = DataGenerator(
        train_paths, train_labels, 
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    val_gen = DataGenerator(
        val_paths, val_labels,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    test_gen = DataGenerator(
        test_paths, test_labels,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen


def train_model(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("="*70)
    print("NEUROPATHOLOGY DETECTION MODEL TRAINING")
    print("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ Found {len(gpus)} GPU(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting error: {e}")
    else:
        print("\n⚠ No GPU found, training will be slow")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    image_paths, labels, class_names = load_dataset(args.data_dir)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(
        image_paths, labels, 
        batch_size=args.batch_size,
        validation_split=0.15,
        test_split=0.15
    )
    
    # Create model
    print(f"\nBuilding model with {args.base_model} base...")
    model_wrapper = NeuropathologyModel(
        num_classes=len(class_names),
        input_shape=(224, 224, 3),
        base_model=args.base_model,
        trainable_layers=args.trainable_layers
    )
    
    model = model_wrapper.build_model()
    model_wrapper.compile_model(learning_rate=args.learning_rate, optimizer='adam')
    
    # Print model summary
    print("\nModel Architecture:")
    print(model_wrapper.get_model_summary())
    
    # Create callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    callbacks = create_callbacks(model_path='models/best_model.h5')
    
    # Stage 1: Train with frozen base
    print("\n" + "="*70)
    print("STAGE 1: FEATURE EXTRACTION (Frozen Base Model)")
    print("="*70)
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_stage1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 2: Fine-tuning
    if args.fine_tune:
        print("\n" + "="*70)
        print("STAGE 2: FINE-TUNING (Unfrozen Layers)")
        print("="*70)
        
        model_wrapper.fine_tune_model(learning_rate=args.learning_rate_finetune)
        
        callbacks = create_callbacks(model_path='models/best_model_finetuned.h5')
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs_stage2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in history1.history.keys():
            history1.history[key].extend(history2.history[key])
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate_model(model, test_gen, class_names)
    evaluator.print_evaluation_report(evaluation)
    
    # Save final model
    final_model_path = 'models/final_model.h5'
    model_wrapper.save_model(final_model_path)
    print(f"\n✓ Model saved to {final_model_path}")
    
    # Plot training history
    print("\nGenerating training plots...")
    visualizer = Visualizer()
    fig = visualizer.plot_training_history(
        history1.history,
        save_path='models/training_history.png'
    )
    print("✓ Training history saved to models/training_history.png")
    
    # Plot confusion matrix
    cm_fig = visualizer.plot_confusion_matrix(
        evaluation['y_true'],
        evaluation['y_pred'],
        class_names,
        save_path='models/confusion_matrix.png'
    )
    print("✓ Confusion matrix saved to models/confusion_matrix.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Model: {final_model_path}")
    print(f"Best Model: models/best_model_finetuned.h5" if args.fine_tune 
          else "models/best_model.h5")
    print(f"Test Accuracy: {evaluation['accuracy']:.4f}")


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train neuropathology detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python train.py --data_dir ./brain_mri_data --epochs_stage1 30 --epochs_stage2 20
    python train.py --data_dir ./data --batch_size 16 --base_model resnet
        """
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'vgg', 'mobilenet'],
                       help='Base model architecture')
    parser.add_argument('--trainable_layers', type=int, default=20,
                       help='Number of layers to fine-tune')
    
    # Training arguments
    parser.add_argument('--epochs_stage1', type=int, default=30,
                       help='Epochs for stage 1 (feature extraction)')
    parser.add_argument('--epochs_stage2', type=int, default=20,
                       help='Epochs for stage 2 (fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for stage 1')
    parser.add_argument('--learning_rate_finetune', type=float, default=0.0001,
                       help='Learning rate for stage 2 (fine-tuning)')
    parser.add_argument('--no_fine_tune', dest='fine_tune', action='store_false',
                       help='Skip fine-tuning stage')
    
    parser.set_defaults(fine_tune=True)
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
