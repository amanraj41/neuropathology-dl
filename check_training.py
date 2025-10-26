#!/usr/bin/env python3
"""
Training Artifacts Verification Script

This script verifies that all required training artifacts are present
and provides a summary of the training status.
"""

import os
import json
import sys
from pathlib import Path

def check_training_artifacts():
    """Check for the presence of training artifacts."""
    
    results = {
        'status': 'incomplete',
        'artifacts': {},
        'warnings': [],
        'errors': []
    }
    
    # Define expected artifacts
    artifacts = {
        'class_names': 'models/class_names.json',
        'metrics': 'models/metrics.json',
        'training_history_plot': 'models/training_history.png',
        'confusion_matrix_plot': 'models/confusion_matrix.png',
        'final_model': 'models/final_model.keras',
        'best_model': 'models/best_model_finetuned.keras',
        'training_log': 'training_log.txt'
    }
    
    # Check each artifact
    for name, path in artifacts.items():
        exists = os.path.exists(path)
        results['artifacts'][name] = {
            'path': path,
            'exists': exists,
            'size': os.path.getsize(path) if exists else 0
        }
        
        if not exists:
            if 'model' in name:
                # Model files are expected to be gitignored
                results['warnings'].append(
                    f"Model file '{path}' not found (expected if gitignored)"
                )
            else:
                results['errors'].append(f"Required file '{path}' not found")
    
    # Load and verify metadata
    if results['artifacts']['class_names']['exists']:
        try:
            with open(artifacts['class_names'], 'r') as f:
                class_names = json.load(f)
                results['class_names'] = class_names
                results['num_classes'] = len(class_names)
        except Exception as e:
            results['errors'].append(f"Error reading class_names.json: {e}")
    
    if results['artifacts']['metrics']['exists']:
        try:
            with open(artifacts['metrics'], 'r') as f:
                metrics = json.load(f)
                results['accuracy'] = metrics.get('accuracy', 'N/A')
        except Exception as e:
            results['errors'].append(f"Error reading metrics.json: {e}")
    
    # Determine overall status
    critical_artifacts = ['class_names', 'metrics', 'training_history_plot', 
                         'confusion_matrix_plot']
    all_critical_present = all(
        results['artifacts'][a]['exists'] for a in critical_artifacts
    )
    
    if all_critical_present and not results['errors']:
        results['status'] = 'complete'
    elif results['errors']:
        results['status'] = 'error'
    else:
        results['status'] = 'partial'
    
    return results

def print_report(results):
    """Print a formatted report of the verification results."""
    
    print("\n" + "="*70)
    print("TRAINING ARTIFACTS VERIFICATION REPORT")
    print("="*70)
    
    print(f"\nOverall Status: {results['status'].upper()}")
    
    if 'num_classes' in results:
        print(f"Number of Classes: {results['num_classes']}")
        print(f"Classes: {', '.join(results['class_names'])}")
    
    if 'accuracy' in results:
        print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    print("\n" + "-"*70)
    print("Artifact Status:")
    print("-"*70)
    
    for name, info in results['artifacts'].items():
        status = "✓" if info['exists'] else "✗"
        size_str = f"{info['size']:,} bytes" if info['exists'] else "missing"
        print(f"{status} {name:25s} - {size_str:15s} ({info['path']})")
    
    if results['warnings']:
        print("\n" + "-"*70)
        print("Warnings:")
        print("-"*70)
        for warning in results['warnings']:
            print(f"⚠ {warning}")
    
    if results['errors']:
        print("\n" + "-"*70)
        print("Errors:")
        print("-"*70)
        for error in results['errors']:
            print(f"✗ {error}")
    
    print("\n" + "="*70)
    
    # Return exit code
    return 0 if results['status'] in ['complete', 'partial'] else 1

if __name__ == '__main__':
    results = check_training_artifacts()
    exit_code = print_report(results)
    sys.exit(exit_code)
