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
        # Glioma variants (T1, T1C+, T2)
        'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1': '''
        <b>Gliomas</b> are primary brain tumors arising from glial cells. This category includes astrocytomas, gangliogliomas, glioblastomas, oligodendrogliomas, and ependymomas.
        <br><b>Symptoms:</b> Headaches, seizures, cognitive decline, focal neurological deficits, nausea/vomiting.
        <br><b>Causes:</b> Genetic mutations, prior radiation exposure, rare hereditary syndromes (e.g., neurofibromatosis, Li-Fraumeni).
        <br><b>Severity:</b> Ranges from low-grade (slow-growing) to high-grade (glioblastoma multiforme; aggressive, poor prognosis).
        <br><b>Treatment:</b> Surgery, radiation, chemotherapy (temozolomide), targeted therapy.
        ''',
        'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+': '''
        <b>Gliomas (T1 post-contrast)</b>: Contrast enhancement helps differentiate high-grade from low-grade gliomas; high-grade lesions show ring or heterogeneous enhancement.
        <br><b>Symptoms:</b> Similar to T1—seizures, headaches, progressive neurological deficits.
        <br><b>Diagnostic Value:</b> T1C+ imaging reveals blood-brain barrier breakdown, necrosis (glioblastoma hallmark), and vascular proliferation.
        <br><b>Severity:</b> Enhancement pattern correlates with grade; ring enhancement often indicates glioblastoma (WHO Grade IV).
        ''',
        'Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2': '''
        <b>Gliomas (T2-weighted)</b>: T2 highlights edema and tumor extent. Low-grade gliomas appear hyperintense without enhancement; high-grade show heterogeneous signal.
        <br><b>Symptoms:</b> Headaches, seizures, focal weakness, speech/vision disturbances.
        <br><b>Diagnostic Value:</b> Best for detecting tumor infiltration and surrounding vasogenic edema.
        <br><b>Severity:</b> Extensive T2 hyperintensity suggests infiltrative growth; poor demarcation indicates aggressive behavior.
        ''',
        
        # Meningioma variants (T1, T1C+, T2)
        'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1': '''
        <b>Meningiomas</b> are typically benign, extra-axial tumors arising from arachnoid cap cells. Grading: low-grade (WHO I), atypical (WHO II), anaplastic (WHO III), and transitional subtypes.
        <br><b>Symptoms:</b> Headaches, seizures, cranial nerve deficits, visual changes, personality changes (if frontal lobe compression).
        <br><b>Causes:</b> Prior radiation, neurofibromatosis type 2 (NF2), hormonal factors (more common in women).
        <br><b>Severity:</b> Most are benign; atypical/anaplastic subtypes have higher recurrence and invasion risk.
        <br><b>Treatment:</b> Surgical resection; radiation for incomplete resection or high-grade.
        ''',
        'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+': '''
        <b>Meningiomas (T1 post-contrast)</b>: Classically show intense, homogeneous enhancement with "dural tail" sign.
        <br><b>Symptoms:</b> Similar to T1—compression symptoms, seizures, headaches.
        <br><b>Diagnostic Value:</b> Enhancement pattern confirms extra-axial location and vascularity; helps distinguish from other lesions.
        <br><b>Severity:</b> Homogeneous enhancement typical of benign; heterogeneous or brain invasion suggests atypical/anaplastic grades.
        ''',
        'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2': '''
        <b>Meningiomas (T2-weighted)</b>: Variable signal (iso- to hyperintense). Shows adjacent brain edema in larger lesions.
        <br><b>Symptoms:</b> Mass effect, seizures, focal deficits depending on location.
        <br><b>Diagnostic Value:</b> T2 reveals peritumoral edema and helps assess brain parenchyma involvement.
        <br><b>Severity:</b> Extensive edema may indicate aggressive behavior or venous compromise.
        ''',
        
        # Normal brain (T1, T2)
        'NORMAL T1': '''
        <b>Normal Brain (T1-weighted)</b>: No mass lesions, hemorrhage, or abnormal enhancement detected. Gray-white matter differentiation preserved.
        <br><b>Clinical Significance:</b> Reassuring imaging; patient symptoms may be non-structural (e.g., migraine, functional disorders).
        <br><b>Follow-up:</b> Clinical correlation recommended; repeat imaging if symptoms progress.
        ''',
        'NORMAL T2': '''
        <b>Normal Brain (T2-weighted)</b>: No abnormal hyperintensity, mass effect, or edema. Ventricles and sulci within normal limits.
        <br><b>Clinical Significance:</b> Normal structural anatomy; excludes common pathologies (tumors, infarcts, demyelination).
        <br><b>Follow-up:</b> Symptoms warrant clinical evaluation; imaging may be repeated if indicated.
        ''',
        
        # Neurocytoma variants (T1, T1C+, T2)
        'Neurocitoma (Central - Intraventricular, Extraventricular) T1': '''
        <b>Neurocytomas</b> are rare, typically benign intraventricular tumors (central neurocytoma) or less commonly extraventricular. Arise from neuronal cells.
        <br><b>Symptoms:</b> Headaches, hydrocephalus (CSF obstruction), seizures, visual disturbances.
        <br><b>Causes:</b> Unknown; sporadic occurrence.
        <br><b>Severity:</b> Benign (WHO Grade II); rare recurrence after gross total resection.
        <br><b>Treatment:</b> Surgical resection; radiation for subtotal resection or recurrence.
        ''',
        'Neurocitoma (Central - Intraventricular, Extraventricular) T1C+': '''
        <b>Neurocytomas (T1 post-contrast)</b>: Variable enhancement (mild to moderate); often heterogeneous with cystic components.
        <br><b>Symptoms:</b> Obstructive hydrocephalus, headaches, nausea/vomiting.
        <br><b>Diagnostic Value:</b> Enhancement pattern and intraventricular location aid diagnosis; "bubbly" or "soap-bubble" appearance.
        <br><b>Severity:</b> Benign but can cause significant symptoms due to CSF obstruction.
        ''',
        'Neurocitoma (Central - Intraventricular, Extraventricular) T2': '''
        <b>Neurocytomas (T2-weighted)</b>: Heterogeneous signal with cysts and calcifications. Iso- to hyperintense.
        <br><b>Symptoms:</b> Headaches, gait disturbances, cognitive changes from hydrocephalus.
        <br><b>Diagnostic Value:</b> T2 highlights cystic components and surrounding edema.
        <br><b>Severity:</b> Benign; good prognosis with complete resection.
        ''',
        
        # Other lesions (T1, T1C+, T2)
        'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1': '''
        <b>Other Lesions</b>: Includes abscesses, cysts, and diverse encephalopathies (infections, metabolic, inflammatory).
        <br><b>Symptoms:</b> Variable—fever, altered mental status, seizures, focal deficits (abscess); headaches, seizures (cyst); encephalopathy symptoms vary widely.
        <br><b>Causes:</b> Infection (bacterial, fungal, parasitic), congenital (arachnoid cyst), metabolic, autoimmune.
        <br><b>Severity:</b> Ranges from benign cysts to life-threatening abscesses or encephalitis.
        <br><b>Treatment:</b> Antibiotics, drainage (abscess); surgical resection (symptomatic cysts); disease-specific for encephalopathies.
        ''',
        'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+': '''
        <b>Other Lesions (T1 post-contrast)</b>: Abscesses show rim enhancement; cysts typically non-enhancing; encephalopathies show variable enhancement.
        <br><b>Symptoms:</b> Fever, headache, seizures (abscess); asymptomatic or headaches (cyst); altered mentation (encephalopathy).
        <br><b>Diagnostic Value:</b> Rim enhancement with restricted diffusion suggests abscess; non-enhancing cyst benign.
        <br><b>Severity:</b> Abscess requires urgent treatment; encephalopathies vary in severity.
        ''',
        'Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2': '''
        <b>Other Lesions (T2-weighted)</b>: Abscesses hyperintense with surrounding edema; cysts hyperintense without edema; encephalopathies show variable T2 changes.
        <br><b>Symptoms:</b> Depends on etiology—fever, seizures, confusion, focal deficits.
        <br><b>Diagnostic Value:</b> T2 highlights edema and lesion extent; DWI confirms abscess (restricted diffusion).
        <br><b>Severity:</b> Urgent workup needed for abscesses and acute encephalopathies.
        ''',
        
        # Schwannoma variants (T1, T1C+, T2)
        'Schwannoma (Acustico, Vestibular - Trigeminal) T1': '''
        <b>Schwannomas</b> are benign nerve sheath tumors, commonly affecting cranial nerve VIII (acoustic/vestibular schwannoma) or V (trigeminal).
        <br><b>Symptoms:</b> Hearing loss, tinnitus, balance problems (acoustic); facial numbness, pain (trigeminal).
        <br><b>Causes:</b> Sporadic or associated with neurofibromatosis type 2 (bilateral acoustic schwannomas).
        <br><b>Severity:</b> Benign; slow-growing; can compress brainstem or cranial nerves if large.
        <br><b>Treatment:</b> Observation (small), surgical resection, stereotactic radiosurgery (Gamma Knife).
        ''',
        'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+': '''
        <b>Schwannomas (T1 post-contrast)</b>: Intense, homogeneous enhancement; classic "ice cream cone" appearance in cerebellopontine angle (acoustic).
        <br><b>Symptoms:</b> Progressive hearing loss, tinnitus, facial weakness/numbness.
        <br><b>Diagnostic Value:</b> Enhancement confirms diagnosis; "ice cream cone" or "trumpet" shape pathognomonic for acoustic schwannoma.
        <br><b>Severity:</b> Benign; larger lesions compress brainstem or cranial nerves, causing significant morbidity.
        ''',
        'Schwannoma (Acustico, Vestibular - Trigeminal) T2': '''
        <b>Schwannomas (T2-weighted)</b>: Hyperintense with possible cystic components. Shows internal architecture (Antoni A/B areas).
        <br><b>Symptoms:</b> Hearing loss, tinnitus, imbalance, facial symptoms.
        <br><b>Diagnostic Value:</b> T2 highlights tumor extent and brainstem compression.
        <br><b>Severity:</b> Benign; treatment needed for symptomatic or growing lesions.
        '''
    }


def get_mri_findings() -> Dict[str, List[str]]:
    """
    MRI hallmark signs and imaging characteristics for each class.
    These are concise, commonly-reported patterns clinicians look for.

    Returns:
        Dictionary mapping class name -> list of imaging findings
    """
    return {
        'Glioma': [
            'Intra-axial mass with surrounding vasogenic edema and mass effect',
            'Heterogeneous or ring enhancement post-contrast (especially high-grade)',
            'Infiltrative margins crossing white matter tracts; possible necrotic core',
            'T2/FLAIR hyperintensity with irregular borders'
        ],
        'Meningioma': [
            'Extra-axial, well-circumscribed dural-based mass',
            'Homogeneous strong enhancement; classic “dural tail” sign',
            'Broad-based dural attachment, often causing hyperostosis of adjacent skull',
            'CSF cleft sign separating mass from brain parenchyma'
        ],
        'Pituitary Tumor': [
            'Sellar/suprasellar mass arising from pituitary gland',
            'Iso- to hypointense on T1; variable enhancement post-contrast',
            'Upward displacement/compression of optic chiasm in macroadenomas',
            'Cavernous sinus invasion in larger lesions'
        ],
        'Normal': [
            'No mass effect or midline shift',
            'Preserved gray–white matter differentiation',
            'No abnormal enhancement or restricted diffusion',
            'Ventricular system and cisterns within normal limits'
        ]
    }
