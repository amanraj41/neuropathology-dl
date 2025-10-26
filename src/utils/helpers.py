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
    Get the list of neuropathology class names for the 17-class Kaggle dataset.
    
    Returns:
        List of 17 class names across multiple MRI modalities (T1, T1C+, T2)
    
    Note: These classes are specific to the Kaggle Brain MRI 17-Class Dataset.
    If using a different dataset, update this function to match your classes.
    """
    # 17-class dataset (Kaggle brain MRI)
    return [
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1",
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+",
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2",
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1",
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+",
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2",
        "NORMAL T1",
        "NORMAL T2",
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1",
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+",
        "Neurocitoma (Central - Intraventricular, Extraventricular) T2",
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1",
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+",
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2",
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1",
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+",
        "Schwannoma (Acustico, Vestibular - Trigeminal) T2"
    ]


def get_class_descriptions() -> Dict[str, str]:
    """
    Get detailed descriptions of each pathology class.
    
    Returns:
        Dictionary mapping class names to clinical descriptions
    """
    # 17-class dataset: comprehensive clinical info for each class
    return {
        # Glioma variants
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1": """
        <p><b>Gliomas</b> are primary brain tumors arising from glial cells (astrocytes, oligodendrocytes, ependymal cells). They represent approximately 30% of all brain tumors and 80% of malignant brain tumors.</p>
        <p><b>Clinical Presentation:</b> Headaches (often worse in morning), seizures, progressive neurological deficits, cognitive/personality changes, nausea/vomiting due to increased intracranial pressure.</p>
        <p><b>Grading:</b> WHO Grade I-IV. Low-grade (I-II) are slow-growing; high-grade (III-IV, including glioblastoma) are aggressive with median survival 12-15 months for GBM despite treatment.</p>
        <p><b>Treatment:</b> Maximal safe surgical resection, radiation therapy (standard fractionation 60 Gy), temozolomide chemotherapy. Tumor-treating fields for glioblastoma. Clinical trials for recurrent disease.</p>
        """,
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+": """
        <p><b>Gliomas (T1 post-contrast imaging)</b> show enhancement patterns that help grade tumors and plan treatment. Contrast enhancement indicates blood-brain barrier breakdown.</p>
        <p><b>Enhancement Patterns:</b> Ring enhancement with necrotic core suggests glioblastoma (WHO Grade IV). Heterogeneous enhancement indicates anaplastic glioma (Grade III). Minimal/no enhancement suggests low-grade glioma (Grade I-II).</p>
        <p><b>Diagnostic Value:</b> Helps differentiate high-grade from low-grade lesions. Guides biopsy targeting to most aggressive areas. Monitors treatment response and detects recurrence.</p>
        """,
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2": """
        <p><b>Gliomas (T2-weighted imaging)</b> show hyperintense signal that better delineates tumor extent and peritumoral edema compared to T1.</p>
        <p><b>T2/FLAIR Significance:</b> Demonstrates infiltrative tumor margins extending beyond enhancing component. Extensive hyperintensity suggests aggressive, infiltrative growth pattern. Helps surgical planning by showing full tumor extent.</p>
        <p><b>Prognostic Value:</b> T2/FLAIR mismatch sign in IDH-mutant, 1p/19q non-codeleted astrocytomas. Large T2 volume correlates with worse prognosis.</p>
        """,
        # Meningioma variants
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1": """
        <p><b>Meningiomas</b> are typically benign tumors arising from arachnoid cap cells of the meninges. They account for ~38% of all primary brain tumors and are more common in women (2:1 ratio).</p>
        <p><b>Clinical Presentation:</b> Often asymptomatic (incidental finding). When symptomatic: headaches, seizures, focal neurological deficits depending on location, cranial nerve palsies, visual disturbances.</p>
        <p><b>Grading:</b> WHO Grade I (benign, 80-85%), Grade II (atypical, 15-20%), Grade III (anaplastic/malignant, 1-3%). Higher grades have increased recurrence rates.</p>
        <p><b>Treatment:</b> Observation for small, asymptomatic lesions. Surgical resection for symptomatic or growing tumors (Simpson grading system). Radiation for residual/recurrent disease or inoperable tumors.</p>
        """,
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+": """
        <p><b>Meningiomas (T1 post-contrast)</b> show characteristic intense, homogeneous enhancement that is pathognomonic for this tumor type.</p>
        <p><b>Dural Tail Sign:</b> Enhancement along adjacent dura extending from tumor base (~60-70% of cases). Highly suggestive but not specific for meningioma. Represents tumor infiltration or reactive dural thickening.</p>
        <p><b>Diagnostic Certainty:</b> Combination of extra-axial location, broad dural base, homogeneous enhancement, and dural tail provides ~95% diagnostic accuracy pre-operatively.</p>
        """,
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2": """
        <p><b>Meningiomas (T2-weighted)</b> show variable signal intensity that correlates with histologic subtype and surrounding brain edema.</p>
        <p><b>Signal Patterns:</b> Meningothelial/transitional types: isointense. Fibrous types: hypointense. Angiomatous types: hyperintense with flow voids. Extensive peritumoral edema may indicate aggressive behavior or venous obstruction.</p>
        <p><b>Surgical Planning:</b> T2 helps assess brain-tumor interface, predict ease of dissection, and identify venous sinus involvement.</p>
        """,
        # Normal brain
        "NORMAL T1": """
        <p><b>Normal Brain Anatomy (T1-weighted)</b> shows excellent gray-white matter differentiation with no evidence of mass lesions, abnormal enhancement, or structural abnormalities.</p>
        <p><b>Clinical Significance:</b> Excludes structural causes of neurological symptoms. Patient symptoms may be functional, metabolic, or related to conditions not visible on conventional MRI (e.g., early neurodegenerative disease, psychiatric disorders).</p>
        <p><b>Follow-up:</b> Consider clinical correlation, EEG, lumbar puncture, advanced imaging (fMRI, MRS, PET), or repeat imaging if symptoms progress.</p>
        """,
        "NORMAL T2": """
        <p><b>Normal Brain Anatomy (T2-weighted)</b> demonstrates normal CSF spaces, absence of edema, and no abnormal hyperintensity suggesting inflammation or ischemia.</p>
        <p><b>Exclusions:</b> Rules out common pathologies including stroke, multiple sclerosis, tumors with edema, infections, and inflammatory conditions. Normal age-related white matter changes may be seen in older adults.</p>
        """,
        # Neurocytoma variants
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1": """
        <p><b>Neurocytomas</b> are rare (0.1-0.5% of intracranial tumors), benign (WHO Grade II) neuronal tumors typically arising in the lateral ventricles near the foramen of Monro.</p>
        <p><b>Demographics:</b> Young adults (20-40 years), slight male predominance. Central neurocytoma is the classic variant; extraventricular forms are rare.</p>
        <p><b>Clinical Presentation:</b> Headaches, nausea/vomiting from obstructive hydrocephalus, seizures, cognitive changes, rarely intraventricular hemorrhage.</p>
        <p><b>Treatment & Prognosis:</b> Gross total resection is curative (>90% 5-year survival). Subtotal resection followed by radiation. Low recurrence rate. Excellent long-term prognosis.</p>
        """,
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+": """
        <p><b>Neurocytomas (T1 post-contrast)</b> typically show heterogeneous enhancement with areas of calcification and cystic change.</p>
        <p><b>Enhancement Pattern:</b> Variable enhancement (50-90% of cases). Helps delineate solid tumor components from cysts. Identifies vascular supply for surgical planning.</p>
        """,
        "Neurocitoma (Central - Intraventricular, Extraventricular) T2": """
        <p><b>Neurocytomas (T2-weighted)</b> appear as hyperintense intraventricular masses, often with 'soap bubble' appearance due to cysts and calcifications.</p>
        <p><b>Imaging Features:</b> Intraventricular location (attached to septum pellucidum), calcifications (50-70%), cystic components, hydrocephalus. Differential includes subependymoma, ependymoma, and choroid plexus papilloma.</p>
        """,
        # Outros Tipos de Lesões
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1": """
        <p><b>Other Lesions Category</b> encompasses diverse pathologies: brain abscesses (bacterial/fungal), arachnoid/epidermoid cysts, and various encephalopathies (metabolic, toxic, infectious).</p>
        <p><b>Brain Abscess:</b> Focal infection with pus-filled cavity. Causes: bacterial (Staph, Strep), fungal (Aspergillus), parasitic. Risk factors: immunosuppression, endocarditis, dental infections. Treatment: antibiotics ± surgical drainage. Mortality 10-20%.</p>
        <p><b>Cysts:</b> Benign, CSF-filled cavities. Arachnoid cysts are developmental. Epidermoid cysts contain keratin. Usually asymptomatic, treatment only if symptomatic.</p>
        <p><b>Encephalopathies:</b> Diffuse brain dysfunction from metabolic (hepatic, uremic), toxic (drugs, alcohol), or infectious (HIV, PML) causes. Treatment addresses underlying etiology.</p>
        """,
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+": """
        <p><b>Other Lesions (T1 post-contrast)</b> show distinct enhancement patterns that help differentiate between abscess, cyst, and encephalopathy.</p>
        <p><b>Abscess:</b> Classic 'ring enhancement' with smooth, thin wall. Restricted diffusion on DWI confirms pus (differentiates from necrotic tumor with thicker, irregular wall).</p>
        <p><b>Cyst:</b> No enhancement (benign, CSF-filled). Epidermoid shows subtle rim enhancement.</p>
        <p><b>Encephalopathy:</b> Variable patterns. Toxic leukoencephalopathy: white matter enhancement. PRES: cortical/subcortical enhancement.</p>
        """,
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2": """
        <p><b>Other Lesions (T2-weighted)</b> demonstrate characteristic patterns for each entity.</p>
        <p><b>Cyst:</b> Homogeneous hyperintense signal (follows CSF). Arachnoid cyst: CSF-intensity on all sequences. Epidermoid: slightly hyperintense to CSF ('dirty' appearance), restricted diffusion.</p>
        <p><b>Abscess:</b> Central hyperintensity (pus), surrounding vasogenic edema, mass effect. DWI restriction key differentiator.</p>
        <p><b>Encephalopathy:</b> Diffuse T2/FLAIR hyperintensity. Patterns vary: symmetric (metabolic), asymmetric (toxic), posterior (PRES).</p>
        """,
        # Schwannoma variants
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1": """
        <p><b>Schwannomas</b> are benign (WHO Grade I) nerve sheath tumors arising from Schwann cells. Acoustic/vestibular schwannomas (cranial nerve VIII) are most common intracranial type (6-8% of intracranial tumors).</p>
        <p><b>Clinical Presentation:</b> Unilateral hearing loss (most common, gradual onset), tinnitus, vertigo/imbalance, facial numbness/weakness (with large tumors compressing CN V/VII), rarely brainstem compression.</p>
        <p><b>Natural History:</b> Growth rate 1-2 mm/year. 50% show no growth over 5 years. Bilateral acoustic schwannomas pathognomonic for Neurofibromatosis type 2 (NF2).</p>
        <p><b>Treatment Options:</b> Observation for small (<1.5 cm), asymptomatic tumors. Microsurgical resection (retrosigmoid, translabyrinthine, middle fossa approach). Stereotactic radiosurgery (12-13 Gy marginal dose) for tumors <3 cm. Goal: hearing preservation, facial nerve preservation.</p>
        """,
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+": """
        <p><b>Schwannomas (T1 post-contrast)</b> show intense, often homogeneous enhancement that clearly delineates the tumor and its relationship to adjacent neural structures.</p>
        <p><b>Enhancement Pattern:</b> Intense homogeneous enhancement (small tumors). Heterogeneous with cystic/necrotic areas (large tumors >3 cm). 'Ice cream cone' sign: enhancing mass in IAC extending into CPA cistern.</p>
        <p><b>Surgical Planning:</b> Contrast imaging essential for: defining tumor extent, assessing internal auditory canal involvement, identifying vascular relationships (AICA), planning surgical approach.</p>
        """,
        "Schwannoma (Acustico, Vestibular - Trigeminal) T2": """
        <p><b>Schwannomas (T2-weighted)</b> typically appear hyperintense with variable signal reflecting cystic degeneration in larger tumors.</p>
        <p><b>Imaging Features:</b> Hyperintense mass along CN VIII course. Enlargement of internal auditory canal. Large tumors show heterogeneous signal (cystic change, hemorrhage). Brainstem compression visible with large tumors (>3 cm). CSF flow voids may be seen.</p>
        <p><b>Differential Diagnosis:</b> Meningioma (dural tail, broad base), epidermoid cyst (restricted diffusion, 'engulfs' vessels), arachnoid cyst (CSF-intensity, no enhancement).</p>
        """
    }


def get_mri_findings() -> Dict[str, List[str]]:
    """
    MRI hallmark signs and imaging characteristics for each class.
    These are concise, commonly-reported patterns clinicians look for.

    Returns:
        Dictionary mapping class name -> list of imaging findings
    """
    # 17-class dataset: MRI findings for each class
    return {
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1": [
            "Intra-axial mass, hypointense on T1",
            "Mass effect, midline shift possible",
            "Variable enhancement depending on grade"
        ],
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+": [
            "Ring or heterogeneous enhancement post-contrast",
            "Blood-brain barrier breakdown",
            "Necrotic core in high-grade lesions"
        ],
        "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2": [
            "Hyperintense signal on T2/FLAIR",
            "Extensive peritumoral edema",
            "Infiltrative margins"
        ],
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1": [
            "Extra-axial, isointense to cortex",
            "Well-circumscribed mass",
            "Possible calcifications"
        ],
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+": [
            "Intense, homogeneous enhancement",
            "Dural tail sign",
            "Broad-based dural attachment"
        ],
        "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2": [
            "Variable signal (iso- to hyperintense)",
            "Adjacent brain edema",
            "Mass effect"
        ],
        "NORMAL T1": [
            "No mass lesions",
            "Preserved gray-white matter differentiation",
            "No abnormal enhancement"
        ],
        "NORMAL T2": [
            "No abnormal hyperintensity",
            "Normal ventricles and sulci",
            "No edema or mass effect"
        ],
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1": [
            "Intraventricular mass, isointense to cortex",
            "Well-defined margins",
            "Possible calcifications"
        ],
        "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+": [
            "Enhancement delineates tumor margins",
            "Vascularity assessment"
        ],
        "Neurocitoma (Central - Intraventricular, Extraventricular) T2": [
            "Hyperintense mass",
            "Cystic changes or calcifications",
            "Hydrocephalus if obstructive"
        ],
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1": [
            "Abscess: hypointense lesion, mass effect",
            "Cyst: well-circumscribed, CSF-like signal",
            "Encephalopathy: diffuse changes"
        ],
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+": [
            "Abscess: ring enhancement",
            "Cyst: no enhancement",
            "Encephalopathy: diffuse enhancement"
        ],
        "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2": [
            "Cyst: hyperintense signal",
            "Abscess: surrounding edema",
            "Encephalopathy: diffuse T2 changes"
        ],
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1": [
            "Iso- to hypointense mass along nerve course",
            "Well-circumscribed",
            "Possible cystic changes"
        ],
        "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+": [
            "Intense enhancement delineates tumor margins",
            "Nerve involvement assessment"
        ],
        "Schwannoma (Acustico, Vestibular - Trigeminal) T2": [
            "Hyperintense mass",
            "Cystic changes or peritumoral edema",
            "Facial nerve involvement"
        ]
    }
