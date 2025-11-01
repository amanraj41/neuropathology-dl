#!/usr/bin/env python3
"""
Evaluate saved models on the test split and write per-model metrics.

- Reuses train.py's dataset loading and split (random_state=42) to ensure consistency
- Evaluates models/best_model.keras and models/final_model.keras if present
- Writes metrics to models/metrics_<model_basename>.json

Usage:
  python scripts/evaluate_models.py --data_dir ./data/brain_mri_17 --batch_size 24

"""
import os
import sys
import json
import argparse

# Ensure src is on path and we can import train.py helpers
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'src'))

from train import load_dataset, create_data_generators  # type: ignore
from src.utils.helpers import ModelEvaluator
from src.models.neuropathology_model import NeuropathologyModel


def evaluate_and_save(model_path: str, data_dir: str, batch_size: int = 24) -> float:
    """Load model, build test split, evaluate, and save metrics JSON.

    Returns: accuracy as float
    """
    if not os.path.exists(model_path):
        print(f"[Skip] Model not found: {model_path}")
        return -1.0

    # Load dataset and build generators with same split params as training
    image_paths, labels_onehot, class_names = load_dataset(data_dir)
    _, _, test_gen = create_data_generators(
        image_paths, labels_onehot, batch_size=batch_size,
        validation_split=0.15, test_split=0.15,
    )

    # Load model
    wrapper = NeuropathologyModel()
    wrapper.load_model(model_path)
    model = wrapper.model

    # Evaluate
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate_model(model, test_gen, class_names)

    # Save per-model metrics JSON
    metrics_out = {
        'accuracy': float(evaluation['accuracy']),
        'classification_report': evaluation['classification_report']
    }
    os.makedirs('models', exist_ok=True)
    base = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join('models', f'metrics_{base}.json')
    with open(out_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"✓ Saved metrics to {out_path} (accuracy={metrics_out['accuracy']:.4f})")
    return metrics_out['accuracy']


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved models and write per-model metrics JSON.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for evaluation')
    parser.add_argument('--models', nargs='*', default=['models/best_model.keras', 'models/final_model.keras'],
                        help='List of model paths to evaluate')
    args = parser.parse_args()

    # Run evaluations
    results = {}
    for m in args.models:
        acc = evaluate_and_save(m, args.data_dir, args.batch_size)
        if acc >= 0:
            results[m] = acc

    if results:
        print("\nSummary:")
        for m, acc in results.items():
            print(f"  {m}: {acc:.4f}")
    else:
        print("No models were evaluated.")


if __name__ == '__main__':
    main()
