"""
Utility Functions for Neuropathology Detection System

This module provides helper functions for visualization, metrics, and model utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px


class Visualizer:
    """
    Visualization utilities for model predictions and training metrics.
    
    Theory - Why Visualization Matters:
    - Helps identify overfitting (train/val loss divergence)
    - Shows which classes model struggles with (confusion matrix)
    - Builds trust in predictions (show confidence scores)
    - Debugging tool for model behavior
    """
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: str = None) -> None:
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history: Training history from model.fit()
            save_path: Optional path to save the plot
        
        Theory - Training Curves:
        - Training loss should decrease over time
        - Validation loss should track training loss
        - Gap between them indicates overfitting
        - Oscillations suggest learning rate too high
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history.get('accuracy', []), label='Training Accuracy')
        axes[0, 0].plot(history.get('val_accuracy', []), label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history.get('loss', []), label='Training Loss')
        axes[0, 1].plot(history.get('val_loss', []), label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if 'precision' in history:
            axes[1, 0].plot(history.get('precision', []), label='Training Precision')
            axes[1, 0].plot(history.get('val_precision', []), label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall
        if 'recall' in history:
            axes[1, 1].plot(history.get('recall', []), label='Training Recall')
            axes[1, 1].plot(history.get('val_recall', []), label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            class_names: List[str],
                            save_path: str = None) -> None:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Optional path to save the plot
        
        Theory - Confusion Matrix:
        - Rows: True labels
        - Columns: Predicted labels
        - Diagonal: Correct predictions
        - Off-diagonal: Errors
        - Helps identify which classes are confused
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_prediction_confidence(predictions: np.ndarray,
                                  class_names: List[str],
                                  top_k: int = 5) -> go.Figure:
        """
        Create an interactive bar chart of prediction confidence.
        
        Args:
            predictions: Probability distribution over classes
            class_names: Names of classes
            top_k: Number of top predictions to show
        
        Returns:
            Plotly figure
        """
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = predictions[top_indices]
        top_classes = [class_names[i] for i in top_indices]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=top_classes, y=top_probs,
                  marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Prediction Confidence',
            xaxis_title='Class',
            yaxis_title='Probability',
            yaxis_range=[0, 1]
        )
        
        return fig


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    
    Theory - Model Evaluation:
    - Training accuracy alone is not enough
    - Must evaluate on unseen test data
    - Multiple metrics give complete picture
    - Class-wise metrics show where model struggles
    """
    
    @staticmethod
    def evaluate_model(model: tf.keras.Model,
                      test_data: tf.keras.utils.Sequence,
                      class_names: List[str]) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained Keras model
            test_data: Test data generator
            class_names: Names of classes
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_true = []
        y_pred = []
        
        for batch_x, batch_y in test_data:
            predictions = model.predict(batch_x)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(batch_y, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, 
                                      target_names=class_names,
                                      output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    @staticmethod
    def print_evaluation_report(evaluation: Dict) -> None:
        """
        Print formatted evaluation report.
        
        Args:
            evaluation: Dictionary from evaluate_model()
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {evaluation['accuracy']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-"*60)
        
        report = evaluation['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\n{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1-score']:.4f}")
                print(f"  Support:   {metrics['support']}")


def create_sample_data(num_samples: int = 100,
                       num_classes: int = 4,
                       img_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic sample data for testing.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        img_size: Image size
    
    Returns:
        Tuple of (images, labels)
    
    Note: This is for demonstration/testing only.
    Real medical data should be used for actual training.
    """
    # Generate random images
    images = np.random.rand(num_samples, img_size[0], img_size[1], 3).astype(np.float32)
    
    # Generate random labels (one-hot encoded)
    labels = np.zeros((num_samples, num_classes))
    random_indices = np.random.randint(0, num_classes, num_samples)
    labels[np.arange(num_samples), random_indices] = 1
    
    return images, labels


def get_class_names() -> List[str]:
    """
    Get the list of neuropathology class names.
    
    Returns:
        List of class names
    
    Note: These are example classes. Actual classes depend on your dataset.
    Common neuropathological conditions:
    - Glioma: Brain tumor arising from glial cells
    - Meningioma: Tumor of meninges (brain/spinal cord membranes)
    - Pituitary tumor: Tumor of pituitary gland
    - Normal: No pathological findings
    """
    return [
        'Glioma',
        'Meningioma',
        'Pituitary Tumor',
        'Normal'
    ]


def get_class_descriptions() -> Dict[str, str]:
    """
    Get detailed descriptions of each pathology class.
    
    Returns:
        Dictionary mapping class names to descriptions
    """
    return {
        'Glioma': '''
        Gliomas are brain tumors that originate from glial cells, which support and protect neurons.
        They are the most common type of brain tumor and can be benign or malignant.
        Symptoms include headaches, seizures, and neurological deficits.
        Treatment may involve surgery, radiation, and chemotherapy.
        ''',
        
        'Meningioma': '''
        Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord.
        Most are benign and slow-growing. They are more common in women.
        Symptoms depend on location but may include headaches, vision problems, and seizures.
        Treatment typically involves surgical removal if symptomatic.
        ''',
        
        'Pituitary Tumor': '''
        Pituitary tumors develop in the pituitary gland at the base of the brain.
        Most are benign adenomas. They can affect hormone production.
        Symptoms include vision changes, headaches, and hormonal imbalances.
        Treatment options include medication, surgery, and radiation therapy.
        ''',
        
        'Normal': '''
        No pathological findings detected in the MRI scan.
        Brain structures appear normal with no signs of tumors or abnormalities.
        Regular monitoring may still be recommended based on symptoms.
        '''
    }
