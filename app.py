"""
Streamlit Web Application for Neuropathology Detection

A modern, interactive web interface for brain MRI analysis using deep learning.

This application demonstrates:
- Deep learning model deployment
- Interactive medical image analysis
- Real-time prediction visualization

"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from io import BytesIO

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel, create_callbacks
from src.data.data_loader import MRIDataLoader
from src.utils.helpers import (
    Visualizer, ModelEvaluator, get_class_names, 
    get_class_descriptions, get_mri_findings, create_sample_data
)

# Page configuration
st.set_page_config(
    page_title="Neuropathology Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #145a8a;
    }
</style>
""", unsafe_allow_html=True)


class NeuropathologyApp:
    """
    Main application class for the Streamlit interface.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.class_names = get_class_names()
        self.class_descriptions = get_class_descriptions()
        self.class_mri_findings = get_mri_findings()
        self.data_loader = MRIDataLoader(img_size=(224, 224))
        
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        # Try to load trained model if available
        final_model_path = os.path.join('models', 'final_model.keras')
        if not st.session_state.model_loaded and os.path.exists(final_model_path):
            try:
                wrapper = NeuropathologyModel()
                wrapper.load_model(final_model_path)
                st.session_state.model = wrapper
                st.session_state.model_loaded = True
                # Try to load class names and metrics for dynamic UI
                class_names_path = os.path.join('models', 'class_names.json')
                metrics_path = os.path.join('models', 'metrics.json')
                if os.path.exists(class_names_path):
                    import json
                    with open(class_names_path, 'r') as f:
                        saved_names = json.load(f)
                    if isinstance(saved_names, list) and len(saved_names) >= 2:
                        self.class_names = saved_names
                if os.path.exists(metrics_path):
                    import json
                    with open(metrics_path, 'r') as f:
                        st.session_state.metrics = json.load(f)
            except Exception as e:
                st.warning(f"Could not load trained model: {e}. Using demo predictions instead.")
    
    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<h1 class="main-header">🧠 Neuropathology Detection System</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Brain MRI Analysis Using Deep Learning</p>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        page = st.session_state.get('page', 'Home')
        
        if page == 'Home':
            self.render_home()
        elif page == 'Detection':
            self.render_detection()
        elif page == 'About Model':
            self.render_about_model()
        # 'Theory' page removed from navigation/UI
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("Navigation")
        
        pages = ['Home', 'Detection', 'About Model']
        st.session_state.page = st.sidebar.radio("Go to", pages)
        
        st.sidebar.markdown("---")
        
        # Sidebar quick info (dynamic)
        acc_text = None
        if 'metrics' in st.session_state and isinstance(st.session_state.metrics, dict):
            try:
                acc = float(st.session_state.metrics.get('accuracy', None))
                if acc:
                    acc_text = f"**Accuracy**: {acc*100:.2f}%  "
            except Exception:
                pass
        classes_preview = ', '.join(self.class_names[:6])
        more = max(0, len(self.class_names) - 6)
        more_text = f" and {more} more" if more > 0 else ""
        sidebar_md = f"""
        ### 📋 Quick Info

        This system uses **Transfer Learning** with pre-trained deep neural networks.

        **Model**: MobileNetV2  
        **Input**: 224×224 MRI images  
        **Classes**: {len(self.class_names)} (e.g., {classes_preview}{more_text})
        """
        if acc_text:
            sidebar_md += "\n\n" + acc_text
        st.sidebar.markdown(sidebar_md)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        st.session_state.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for predictions"
        )
    
    def render_home(self):
        """Render the home page."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Project Overview</h3>
            <p>
            This comprehensive neuropathology detection system leverages state-of-the-art 
            deep learning techniques to analyze brain MRI scans and detect various pathological 
            conditions. The system combines:
            </p>
            <ul>
                <li><strong>Transfer Learning</strong>: Using pre-trained MobileNetV2 model</li>
                <li><strong>Fine-tuning</strong>: Adapting general features to medical imaging</li>
                <li><strong>Modern UI</strong>: Interactive Streamlit interface for instant diagnosis</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                ### 🚀 Key Features

                1. **Production-Ready DL Pipeline**
                    - Transfer learning from ImageNet pre-trained MobileNetV2
                    - Two-stage training: feature extraction + fine-tuning
                    - Robust callbacks: checkpointing, early stopping, LR scheduling

                2. **Multiple Pathology Detection**
                    - Glioma: Glial cell tumors
                    - Meningioma: Meningeal tumors
                    - Pituitary Tumors: Pituitary gland abnormalities
                    - Normal brain classification

                3. **Interactive Analysis**
                    - Real-time prediction with confidence scores
                    - Detailed pathology descriptions and MRI characteristics
                    - Interactive Plotly visualizations

                4. **Model Implementation Details**
                    - Base: MobileNetV2 (2.26M params, include_top=False)
                    - Head: GAP → BatchNorm → Dense(512, ReLU) → Dropout(0.5)
                      → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(4, Softmax)
                    - Optimizer: Adam (stage 1: 1e-3, fine-tune: 1e-4)
                    - Metrics: Accuracy, Precision, Recall, AUC
                    - Achieved Test Accuracy: 84.08%
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Model Statistics
            """)
            
            # Create metric cards
            # Metrics card values (dynamic if metrics available)
            acc_val = None
            if 'metrics' in st.session_state and isinstance(st.session_state.metrics, dict):
                try:
                    acc_val = f"{float(st.session_state.metrics.get('accuracy', 0))*100:.2f}%"
                except Exception:
                    acc_val = None
            metrics = {
                "Model Type": "MobileNetV2",
                "Parameters": "3.05M total",
                "Input Size": "224×224×3",
                "Classes": str(len(self.class_names)),
                "Accuracy": acc_val or "-",
                "Framework": "TensorFlow/Keras"
            }
            
            for metric, value in metrics.items():
                st.metric(label=metric, value=value)
            
            st.markdown("---")
            
            # Removed learning path/educational content per repository policy
    
    def render_detection(self):
        """Render the detection page."""
        st.markdown("## 🔍 Neuropathology Detection")
        
        st.markdown("""
        <div class="info-box">
        Upload a brain MRI image to detect potential pathologies. The system will analyze 
        the image using deep learning and provide predictions with confidence scores.
        </div>
        """, unsafe_allow_html=True)
        
        # Input sources: file upload or URL
        col_up, col_url = st.columns([1, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Choose a brain MRI image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a brain MRI scan in PNG or JPEG format"
            )
        with col_url:
            image_url = st.text_input(
                "Or paste an image URL (jpg/jpeg/png)",
                placeholder="https://.../brain_mri.jpg"
            )
            fetch_from_url = st.button("🌐 Fetch from URL", use_container_width=True)

        image = None
        src_label = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                src_label = 'Uploaded MRI Scan'
            except Exception as e:
                st.error(f"Could not open uploaded image: {e}")
        elif image_url and fetch_from_url:
            try:
                req = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=10) as resp:
                    data = resp.read()
                image = Image.open(BytesIO(data))
                src_label = 'Fetched MRI Scan'
            except Exception as e:
                st.error(f"Failed to fetch image from URL: {e}")

        if image is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                st.image(image, caption=src_label, use_container_width=True)
                
                # Image info
                st.markdown(f"""
                **Image Details:**
                - Size: {image.size[0]}×{image.size[1]} pixels
                - Mode: {image.mode}
                - Source: {src_label}
                """)
            
            with col2:
                st.markdown("### 🤖 Analysis")
                
                # Analyze button
                if st.button("🔬 Analyze Image", key="analyze"):
                    with st.spinner("Analyzing image with deep learning model..."):
                        # Preprocess image
                        try:
                            # Save temp image for preprocessing
                            temp_path = "/tmp/uploaded_image.jpg"
                            image.convert('RGB').save(temp_path)
                            
                            # Load and preprocess
                            img_array = self.data_loader.load_and_preprocess_image(temp_path)
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Make prediction using trained model if loaded, otherwise demo
                            if st.session_state.model_loaded and st.session_state.model is not None:
                                predictions = st.session_state.model.predict(img_batch)
                            else:
                                predictions = self.predict_with_demo_model(img_batch)
                            
                            # Get predicted class and confidence
                            predicted_class = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class]
                            class_name = self.class_names[predicted_class]
                            
                            # Display results
                            self.display_results(predictions[0], predicted_class, 
                                               confidence, class_name)
                            
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
        
        else:
            st.info("👆 Please upload a brain MRI image to begin analysis")
            
            st.markdown("---")
            if st.session_state.model_loaded:
                st.markdown("""
                <div class="success-box">
                <h4>Trained Model Loaded</h4>
                <p>
                The trained model (<code>models/final_model.keras</code>) is loaded. 
                Upload an image to get real predictions.
                </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 📁 Demo Mode")
                st.markdown("""
                <div class="info-box">
                No trained model available yet. The system will demonstrate with random predictions.
                To use a real trained model, you need to:
                <ol>
                    <li>Obtain a brain MRI dataset (e.g., from Kaggle)</li>
                    <li>Train the model using the provided training script</li>
                    <li>Reload this app; it will auto-load the saved model</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)
    
    def predict_with_demo_model(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Make predictions using a demo model.
        
        In production, this would load and use a trained model.
        For now, it generates realistic-looking predictions for demonstration.
        
        Args:
            img_batch: Batch of preprocessed images
        
        Returns:
            Prediction probabilities
        """
        # Generate realistic-looking predictions
        # In production, replace with: model.predict(img_batch)
        
        # Create probabilities that look realistic
        # One class dominant, others much lower
        probs = np.random.dirichlet(np.ones(4) * 0.5, size=1)
        
        return probs
    
    def display_results(self, predictions: np.ndarray, predicted_class: int,
                       confidence: float, class_name: str):
        """Display prediction results."""
        
        # Main prediction
        if confidence >= st.session_state.confidence_threshold:
            st.markdown(f"""
            <div class="success-box">
            <h3>✅ Detection Result</h3>
            <h2>{class_name}</h2>
            <p><strong>Confidence: {confidence*100:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            <h3>⚠️ Low Confidence Detection</h3>
            <h2>{class_name}</h2>
            <p><strong>Confidence: {confidence*100:.2f}%</strong></p>
            <p>The model's confidence is below the threshold. Consider getting a second opinion.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("### 📊 Detailed Analysis")
        
        # Confidence scores for all classes
        fig = go.Figure(data=[
            go.Bar(
                x=self.class_names,
                y=predictions,
                marker_color=['#28a745' if i == predicted_class else '#6c757d' 
                             for i in range(len(self.class_names))],
                text=[f'{p*100:.1f}%' for p in predictions],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Prediction Confidence for All Classes',
            xaxis_title='Pathology Type',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Class description and characteristic MRI findings
        st.markdown("### 📖 About This Condition")
        mri_points = ''.join([f'<li>{pt}</li>' for pt in self.class_mri_findings.get(class_name, [])])
        desc_text = self.class_descriptions.get(class_name, 'Detailed description unavailable for this class.')
        st.markdown(f"""
        <div class="info-box">
        <h4>{class_name}</h4>
        {desc_text}
        <h5>Characteristic MRI Findings</h5>
        <ul>
        {mri_points}
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # All class probabilities
        st.markdown("### 📋 All Class Probabilities")
        for i, (name, prob) in enumerate(zip(self.class_names, predictions)):
            st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")
    
    def render_about_model(self):
        """Render the about model page."""
        st.markdown("## 🤖 About the Model")
        
        st.markdown("""
        <div class="info-box">
        <h3>Model Architecture</h3>
        <p>
        This system uses <strong>Transfer Learning</strong> with MobileNetV2, 
        an efficient convolutional neural network that uses depthwise separable convolutions 
        for optimal performance on medical imaging tasks.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🏗️ Architecture Details
            
            **Base Model: MobileNetV2**
            - Depthwise separable convolutions; inverted residual blocks
            - Optimized for efficiency with strong feature extraction
            - Pre-trained on ImageNet (1.4M images, 1000 classes)
            
            **Custom Classification Head:**
            1. Global Average Pooling (reduces spatial dimensions)
            2. Batch Normalization (stabilizes training)
            3. Dense Layer (512 units, ReLU activation)
            4. Dropout (0.5, prevents overfitting)
            5. Dense Layer (256 units, ReLU activation)
            6. Batch Normalization
            7. Dropout (0.3)
            8. Output Layer (4 units, Softmax activation)
            
            **Total Parameters:** 3.05 Million
            - Trainable: 0.79 Million
            - Frozen (base): 2.26 Million
            """)
        
        with col2:
            st.markdown("""
            ### 📚 Training Strategy
            
            **Two-Stage Training:**
            
            **Stage 1: Feature Learning**
            - Freeze base model weights
            - Train only classification head
            - Learn task-specific features
            - Epochs: 20-30
            - Learning rate: 0.001
            
            **Stage 2: Fine-Tuning**
            - Unfreeze last 20 layers
            - Fine-tune with lower learning rate
            - Adapt features to medical images
            - Epochs: 10-20
            - Learning rate: 0.0001
            
            **Optimization:**
            - Optimizer: Adam
            - Loss: Categorical Cross-Entropy
            - Metrics: Accuracy, Precision, Recall, AUC
            
            **Regularization:**
            - Dropout: 0.5, 0.3
            - Batch Normalization
            - Data Augmentation (rotation, flip, zoom)
            - Early Stopping (patience=10)
            """)
        
        st.markdown("---")
        
        # Model performance
        st.markdown("### 📈 Performance (Test Set)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h2>84.08%</h2>
            <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h2>85.27%</h2>
            <p>Precision (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h2>84.08%</h2>
            <p>Recall (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h2>83.93%</h2>
            <p>F1-Score (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset info
        st.markdown("""
        ### 📊 Dataset Information
        
        **Recommended Dataset:** Brain MRI Images for Brain Tumor Detection
        
        **Classes:**
        - Glioma: Tumors from glial cells
        - Meningioma: Tumors of meninges
        - Pituitary Tumor: Pituitary gland tumors
        - Normal: No pathological findings
        
        **Data Split:**
        - Training: 70%
        - Validation: 15%
        - Testing: 15%
        
        **Preprocessing:**
        - Resize to 224×224 pixels
        - Normalize to [0, 1]
        - Data augmentation (training only)
        
        **Data Augmentation:**
        - Random rotation: ±20°
        - Random zoom: ±10%
        - Horizontal flip: 50% probability
        """)
    
    # Theory rendering methods removed per request to keep repository focused


def main():
    """Main entry point for the application."""
    app = NeuropathologyApp()
    app.run()


if __name__ == "__main__":
    main()
