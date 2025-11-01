"""
Training Script for Neuropathology Detection Model
Handles data loading, model training (two-stage), evaluation, and saving.

Usage: python train.py --data_dir /path/to/data --epochs 30 --batch_size 32
"""

import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel, create_callbacks
from src.data.data_loader import DataGenerator, MRIDataLoader
from src.utils.helpers import Visualizer, ModelEvaluator, get_class_names


def load_dataset(data_dir: str, img_size=(224, 224)):
    """Load dataset from directory structure (class subdirectories with images)."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    image_paths = []
    labels = []
    class_names = []
    
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


def _get_completed_epochs(csv_path: str) -> int:
    """Return how many epochs have already been completed for a stage.

    Strategy:
    - If the CSV file doesn't exist, return 0.
    - Count non-empty data rows, skipping any header lines.
    - Works with files written by keras.callbacks.CSVLogger(append=True).
    """
    try:
        if not os.path.exists(csv_path):
            return 0
        count = 0
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Skip header lines that start with 'epoch'
                if s.lower().startswith('epoch'):
                    continue
                count += 1
        # Each data row corresponds to one epoch completed
        return count
    except Exception as e:
        print(f"[Warning] Could not read CSV history at {csv_path}: {e}")
        return 0


def _get_last_logged_epoch(csv_path: str) -> int:
    """Return the last epoch index found in a CSVLogger file, or -1 if unavailable.

    This reads the 'epoch' column which CSVLogger writes per row. Useful when
    resuming so we can set initial_epoch = last_epoch + 1.
    """
    try:
        if not os.path.exists(csv_path):
            return -1
        last_epoch = -1
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = None
            for line in f:
                s = line.strip()
                if not s:
                    continue
                cols = [c.strip() for c in s.split(',')]
                if header is None:
                    header = [h.strip().lower() for h in cols]
                    continue
                if 'epoch' in header:
                    try:
                        idx = header.index('epoch')
                        ep = int(float(cols[idx]))
                        if ep > last_epoch:
                            last_epoch = ep
                    except Exception:
                        continue
        return last_epoch
    except Exception as e:
        print(f"[Warning] Could not parse last epoch from {csv_path}: {e}")
        return -1


def _get_best_metric_from_csv(csv_path: str, metric: str = 'val_accuracy') -> float | None:
    """Parse a CSVLogger file and return the best (max) value for a metric.

    Returns None if the file doesn't exist or the metric column is not found.
    Supports both 'val_accuracy' and legacy 'val_acc' column names.
    """
    try:
        if not os.path.exists(csv_path):
            return None
        best = None
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = None
            for line in f:
                s = line.strip()
                if not s:
                    continue
                cols = [c.strip() for c in s.split(',')]
                if header is None:
                    header = [h.strip() for h in cols]
                    continue
                # Find metric index with fallback to common alias
                metric_name = metric
                if metric_name not in header:
                    if metric_name == 'val_accuracy' and 'val_acc' in header:
                        metric_name = 'val_acc'
                    else:
                        return None
                try:
                    idx = header.index(metric_name)
                    val = float(cols[idx])
                    if best is None or val > best:
                        best = val
                except Exception:
                    continue
        return best
    except Exception as e:
        print(f"[Warning] Could not parse best metric from {csv_path}: {e}")
        return None


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
    
    # Create callbacks and persist class names for reproducibility
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.join('logs', 'train'), exist_ok=True)
    try:
        with open(os.path.join('models', 'class_names.json'), 'w') as f:
            json.dump(class_names, f, indent=2)
        print("✓ Saved class names to models/class_names.json")
    except Exception as e:
        print(f"[Warning] Could not save class names: {e}")
    # Use per-stage CSV logs so we can compute initial_epoch reliably
    csv_stage1 = os.path.join('logs', 'train', 'history_stage1.csv')
    csv_stage2 = os.path.join('logs', 'train', 'history_stage2.csv')

    # Derive initial checkpoint threshold to protect best on resume
    initial_threshold_stage1 = None
    if getattr(args, 'resume', False):
        # Prefer stage-specific CSV; fall back to legacy history.csv if available
        initial_threshold_stage1 = _get_best_metric_from_csv(csv_stage1, 'val_accuracy')
        if initial_threshold_stage1 is None:
            legacy_csv = os.path.join('logs', 'train', 'history.csv')
            initial_threshold_stage1 = _get_best_metric_from_csv(legacy_csv, 'val_accuracy')

    callbacks = create_callbacks(
        model_path='models/best_model.keras',
        csv_log_path=csv_stage1,
        initial_value_threshold=initial_threshold_stage1
    )

    # Optionally resume from best checkpoint for Stage 1
    if getattr(args, 'resume', False):
        ckpt_stage1 = 'models/best_model.keras'
        if os.path.exists(ckpt_stage1):
            try:
                model.load_weights(ckpt_stage1)
                print(f"\n↻ Resumed weights from checkpoint: {ckpt_stage1}")
            except Exception as e:
                print(f"[Warning] Could not load checkpoint {ckpt_stage1}: {e}")
        else:
            print(f"[Info] Resume flag set but no checkpoint found at {ckpt_stage1}. Starting fresh.")
    
    # Stage 1: Train with frozen base
    print("\n" + "="*70)
    print("STAGE 1: FEATURE EXTRACTION (Frozen Base Model)")
    print("="*70)
    
    # Determine initial_epoch for Stage 1 if resuming
    initial_epoch_stage1 = 0
    if getattr(args, 'resume', False):
        last_ep_stage1 = _get_last_logged_epoch(csv_stage1)
        if last_ep_stage1 >= 0:
            initial_epoch_stage1 = last_ep_stage1 + 1
        else:
            initial_epoch_stage1 = _get_completed_epochs(csv_stage1)
            if initial_epoch_stage1 == 0:
                legacy_csv = os.path.join('logs', 'train', 'history.csv')
                legacy_completed = _get_completed_epochs(legacy_csv)
                if legacy_completed > 0:
                    print(f"[Info] Stage 1 history not found at {csv_stage1}. Using legacy {legacy_csv} count: {legacy_completed}.")
                    initial_epoch_stage1 = legacy_completed
    if initial_epoch_stage1 > 0:
        print(f"\n↻ Resuming Stage 1 from epoch {initial_epoch_stage1} towards {args.epochs_stage1} total epochs (initial_epoch set).")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs_stage1,
        initial_epoch=initial_epoch_stage1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 2: Fine-tuning
    if args.fine_tune:
        print("\n" + "="*70)
        print("STAGE 2: FINE-TUNING (Unfrozen Layers)")
        print("="*70)
        
        # Decide whether to resume Stage 2 (requires both checkpoint and CSV present)
        ckpt_stage2 = 'models/best_model_finetuned.keras'
        stage2_resume_allowed = False
        if getattr(args, 'resume', False) and os.path.exists(ckpt_stage2) and os.path.exists(csv_stage2):
            stage2_resume_allowed = True

        # If we're not resuming Stage 2, or force flag is set, start from BEST Stage 1 checkpoint
        if (not stage2_resume_allowed) and getattr(args, 'force_stage2_from_best', True):
            best_stage1_path = 'models/best_model.keras'
            if os.path.exists(best_stage1_path):
                try:
                    model.load_weights(best_stage1_path)
                    print(f"\n↻ Preparing Stage 2 from BEST Stage 1 checkpoint: {best_stage1_path}")
                except Exception as e:
                    print(f"[Warning] Could not load best Stage 1 checkpoint {best_stage1_path}: {e}")
            else:
                print(f"[Info] Best Stage 1 checkpoint not found at {best_stage1_path}. Proceeding with current weights.")

        # Unfreeze and compile for fine-tuning
        model_wrapper.fine_tune_model(learning_rate=args.learning_rate_finetune)

        # If starting fresh Stage 2, clear any prior Stage 2 CSV history so we don't mis-compute initial_epoch
        if not stage2_resume_allowed and os.path.exists(csv_stage2):
            try:
                os.remove(csv_stage2)
                print(f"[Info] Starting fresh Stage 2: removed prior history file {csv_stage2}")
            except Exception as e:
                print(f"[Warning] Could not remove prior Stage 2 history {csv_stage2}: {e}")

        # Set checkpoint threshold for Stage 2:
        # - if resuming Stage 2, use best val_accuracy from its CSV (protects against first-epoch overwrite)
        # - if starting fresh, use best Stage 1 val_accuracy so fine-tuned checkpoint only saves when it beats Stage 1
        initial_threshold_stage2 = None
        if stage2_resume_allowed:
            initial_threshold_stage2 = _get_best_metric_from_csv(csv_stage2, 'val_accuracy')
            if initial_threshold_stage2 is None:
                legacy_csv = os.path.join('logs', 'train', 'history.csv')
                initial_threshold_stage2 = _get_best_metric_from_csv(legacy_csv, 'val_accuracy')
        else:
            # Use Stage 1 best as baseline if available
            initial_threshold_stage2 = initial_threshold_stage1

        if initial_threshold_stage2 is not None:
            try:
                print(f"[Info] Stage 2 checkpoint threshold set to {initial_threshold_stage2:.5f} (val_accuracy).")
            except Exception:
                print(f"[Info] Stage 2 checkpoint threshold set (val_accuracy).")

        callbacks = create_callbacks(
            model_path='models/best_model_finetuned.keras',
            csv_log_path=csv_stage2,
            initial_value_threshold=initial_threshold_stage2
        )

        # Optionally resume from fine-tuned checkpoint (only if allowed)
        if stage2_resume_allowed:
            try:
                model.load_weights(ckpt_stage2)
                print(f"\n↻ Resumed Stage 2 weights from checkpoint: {ckpt_stage2}")
            except Exception as e:
                print(f"[Warning] Could not load Stage 2 checkpoint {ckpt_stage2}: {e}")
        else:
            print(f"[Info] Not resuming Stage 2; starting fresh fine-tuning run.")
        
        # Determine initial_epoch for Stage 2 if resuming
        initial_epoch_stage2 = 0
        if stage2_resume_allowed:
            last_ep_stage2 = _get_last_logged_epoch(csv_stage2)
            if last_ep_stage2 >= 0:
                initial_epoch_stage2 = last_ep_stage2 + 1
            else:
                initial_epoch_stage2 = _get_completed_epochs(csv_stage2)
                if initial_epoch_stage2 == 0:
                    legacy_csv = os.path.join('logs', 'train', 'history.csv')
                    legacy_completed = _get_completed_epochs(legacy_csv)
                    if legacy_completed > 0:
                        print(f"[Info] Stage 2 history not found at {csv_stage2}. Using legacy {legacy_csv} count: {legacy_completed}.")
                        initial_epoch_stage2 = legacy_completed
        if initial_epoch_stage2 > 0:
            print(f"\n↻ Resuming Stage 2 from epoch {initial_epoch_stage2} towards {args.epochs_stage2} total epochs (initial_epoch set).")
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs_stage2,
            initial_epoch=initial_epoch_stage2,
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

    # Persist evaluation metrics
    try:
        metrics_out = {
            'accuracy': float(evaluation['accuracy']),
            'classification_report': evaluation['classification_report']
        }
        with open(os.path.join('models', 'metrics.json'), 'w') as f:
            json.dump(metrics_out, f, indent=2)
        print("✓ Saved evaluation metrics to models/metrics.json")
    except Exception as e:
        print(f"[Warning] Could not save metrics.json: {e}")
    
    # Save final model
    final_model_path = 'models/final_model.keras'
    model_wrapper.save_model(final_model_path)
    print(f"\n✓ Model saved to {final_model_path}")
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
    print(f"Best Model: models/best_model_finetuned.keras" if args.fine_tune 
          else "models/best_model.keras")
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
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from best available checkpoint(s)')
    # Stage 2 sourcing flags
    parser.add_argument('--force_stage2_from_best', dest='force_stage2_from_best', action='store_true',
                       help='Before fine-tuning, load best Stage 1 checkpoint (models/best_model.keras).')
    parser.add_argument('--no_force_stage2_from_best', dest='force_stage2_from_best', action='store_false',
                       help='Do not force load best Stage 1 checkpoint; use current weights or resume Stage 2 if available.')
    
    parser.set_defaults(fine_tune=True, force_stage2_from_best=True)
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        parser.error(f"Data directory does not exist: {args.data_dir}")
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
