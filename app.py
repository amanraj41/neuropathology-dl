"""
NeuroPathology AI Diagnostics - Streamlit Web Application
Interactive web interface for brain MRI analysis using deep learning.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import plotly.graph_objects as go
from urllib.request import urlopen, Request
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel
from src.data.data_loader import MRIDataLoader
from src.utils.helpers import get_class_names, get_class_descriptions, get_mri_findings

# Page configuration
st.set_page_config(
    page_title="NeuroPathology AI Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI with color-coded classes
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .title-gradient {
        background: linear-gradient(135deg, #c2185b 0%, #7b1fa2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.35rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .centered-section {
        max-width: 800px;
        margin: 2rem auto;
        text-align: center;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffa726;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #66bb6a;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .class-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    .class-item:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .class-bullet {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 12px;
        flex-shrink: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# Color palette for 17 classes
CLASS_COLORS = {
    "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1": "#e74c3c",
    "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+": "#c0392b",
    "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2": "#d35400",
    "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1": "#3498db",
    "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+": "#2980b9",
    "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2": "#1abc9c",
    "NORMAL T1": "#27ae60",
    "NORMAL T2": "#16a085",
    "Neurocitoma (Central - Intraventricular, Extraventricular) T1": "#9b59b6",
    "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+": "#8e44ad",
    "Neurocitoma (Central - Intraventricular, Extraventricular) T2": "#af7ac5",
    "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1": "#f39c12",
    "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+": "#e67e22",
    "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2": "#d68910",
    "Schwannoma (Acustico, Vestibular - Trigeminal) T1": "#e91e63",
    "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+": "#c2185b",
    "Schwannoma (Acustico, Vestibular - Trigeminal) T2": "#ad1457"
}


class NeuropathologyApp:
    """Main application class for the Streamlit interface."""
    
    def __init__(self):
        self.class_names = get_class_names()
        self.class_descriptions = get_class_descriptions()
        self.class_mri_findings = get_mri_findings()
        self.data_loader = MRIDataLoader(img_size=(224, 224))
        
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'available_models' not in st.session_state:
            st.session_state.available_models = self._discover_models()
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'image_source' not in st.session_state:
            st.session_state.image_source = None
    
    def _discover_models(self):
        models_dir = 'models'
        available = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.keras'):
                    available.append(file)
        return sorted(available)
    
    def _load_model(self, model_name):
        try:
            model_path = os.path.join('models', model_name)
            wrapper = NeuropathologyModel()
            wrapper.load_model(model_path)
            st.session_state.model = wrapper
            st.session_state.model_loaded = True
            st.session_state.loaded_model_name = model_name
            
            # Try to load class names and metrics for dynamic UI
            class_names_path = os.path.join('models', 'class_names.json')
            # Prefer per-model metrics if available, else fall back to global metrics.json
            base = os.path.splitext(model_name)[0]
            metrics_path_specific = os.path.join('models', f'metrics_{base}.json')
            metrics_path = os.path.join('models', 'metrics.json')
            if os.path.exists(class_names_path):
                import json
                with open(class_names_path, 'r') as f:
                    saved_names = json.load(f)
                if isinstance(saved_names, list) and len(saved_names) >= 2:
                    self.class_names = saved_names
            # Load per-model metrics first
            if os.path.exists(metrics_path_specific):
                import json
                with open(metrics_path_specific, 'r') as f:
                    st.session_state.metrics = json.load(f)
            elif os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    st.session_state.metrics = json.load(f)
            # Update descriptions and findings for loaded class names
            self.class_descriptions = get_class_descriptions()
            self.class_mri_findings = get_mri_findings()
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    
    def run(self):
        """Run the main application."""
        # Header with inline emoji and improved title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="main-header" style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="font-size: 3rem;">🧠</span>
                <span class="title-gradient">NeuroPathology AI Diagnostics</span>
            </h1>
            <p class="sub-header" style="font-size: 1.5rem;">Advanced Deep Learning for Precision Brain MRI Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        page = st.session_state.get('page', 'Home')
        
        if page == 'Home':
            self.render_home()
        elif page == 'Detection':
            self.render_detection()
        elif page == 'Diagnosis Classes':
            self.render_diagnosis_classes()
        elif page == 'About Model':
            self.render_about_model()
        # 'Theory' page removed from navigation/UI
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("Navigation")
        
        pages = ['Home', 'Diagnosis Classes', 'Detection', 'About Model']
        new_page = st.sidebar.radio("Go to", pages)
        
        # Only clear analysis results when switching away from Detection page
        # but keep the uploaded image
        if 'page' in st.session_state and st.session_state.page != new_page:
            if new_page != 'Detection' and 'analysis_results' in st.session_state:
                # Keep analysis_results when returning to Detection
                pass
        
        st.session_state.page = new_page
        
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
        
        # Creator information at bottom of sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1.5rem 0.5rem; background: linear-gradient(135deg, #667eea15, #764ba215); border-radius: 10px; margin-top: 1rem;">
            <p style="color: #667eea; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">
                🧠 NeuroPathology AI
            </p>
            <p style="color: #888; font-size: 0.75rem; margin-bottom: 0.5rem;">
                Version 1.0
            </p>
            <p style="color: #666; font-size: 0.8rem; margin-bottom: 0.2rem;">
                <strong>Created by</strong>
            </p>
            <p style="color: #667eea; font-size: 0.95rem; font-weight: 600; margin: 0;">
                Aman Raj
            </p>
            <p style="color: #999; font-size: 0.7rem; margin-top: 0.5rem;">
                © 2025
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_home(self):
        """Render the home page."""
        st.markdown("""
        <div class=\"info-box\" style="text-align: center;">
        <h3>🎯 Project Overview</h3>
    <p style="font-size: 1.1rem; line-height: 1.6;">
    An advanced neuropathology detection system leveraging deep learning to analyze brain MRI scans across multiple imaging modalities (T1, T1C+, T2).<br>
    Provides real-time <strong>17-class</strong> categorization with clear, color‑coded outputs and reference context.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🚀 Key Features")
            st.markdown("""
            - **Deep Learning Framework**: TensorFlow 2.17.0 with Keras 3.4.1
            - **CUDA® Toolkit**: 12.3
            - **cuDNN SDK**: 8.9.7
            - **NVIDIA® GPU Driver**: Version 576.83
            - **Architecture**: MobileNetV2 (ImageNet pre-trained)
            - **Training Strategy**: Two-stage transfer learning with fine-tuning
            - **17-Class Classification**: Covers gliomas, meningiomas, schwannomas, neurocytomas, lesions & normal scans
            - **Data Augmentation**: Rotation, zoom, horizontal flip for robust training
            - **Real-time Inference**: Sub-second predictions on CPU
            - **Interactive Visualizations**: Plotly-powered confidence charts
            - **Medical Context**: Detailed MRI findings and clinical descriptions
            """)
            
            st.markdown("### 🔬 Quick Start")
            st.markdown("""
            1. Navigate to **Diagnosis Classes** to explore all 17 pathology types
            2. Go to **Detection** to analyze MRI scans
            3. Load a trained model from the dropdown menu
            4. Upload an MRI image (or provide a URL)
            5. View color-coded diagnosis results with confidence scores
            """)

        with col2:
            st.markdown("### 📊 Model Statistics")
            # Dynamic metrics
            acc_val = "-"
            if 'metrics' in st.session_state and isinstance(st.session_state.metrics, dict):
                try:
                    acc_val = f"{float(st.session_state.metrics.get('accuracy', 0))*100:.2f}%"
                except Exception:
                    pass

            st.metric(label="Architecture", value="MobileNetV2")
            st.metric(label="Total Parameters", value="3.05M")
            st.metric(label="Trainable Parameters", value="0.79M")
            st.metric(label="Classification Classes", value=str(len(self.class_names)))
            st.metric(label="Test Accuracy", value=acc_val)
            st.metric(label="Input Resolution", value="224×224×3")

    def render_diagnosis_classes(self):
        """Render the diagnosis classes page with color-coded bullets."""
        st.markdown("## 🏥 Diagnosis Classes")

        st.markdown("""
        <div class=\"info-box\">
        <p>
        This system can detect <strong>17 different neuropathological conditions</strong> 
        across multiple MRI imaging modalities (T1, T1C+, T2). Each class is color-coded 
        throughout the interface for easy identification.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Complete Class List")

        # Group classes by type for better organization
        class_groups = {
            "🔴 Gliomas (Malignant Brain Tumors)": [
                "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1",
                "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+",
                "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2"
            ],
            "🔵 Meningiomas (Benign/Atypical Tumors)": [
                "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1",
                "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+",
                "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2"
            ],
            "🟣 Neurocytomas (Rare Neuronal Tumors)": [
                "Neurocitoma (Central - Intraventricular, Extraventricular) T1",
                "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+",
                "Neurocitoma (Central - Intraventricular, Extraventricular) T2"
            ],
            "🟡 Schwannomas (Nerve Sheath Tumors)": [
                "Schwannoma (Acustico, Vestibular - Trigeminal) T1",
                "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+",
                "Schwannoma (Acustico, Vestibular - Trigeminal) T2"
            ],
            "🟠 Other Lesions (Abscesses, Cysts, Encephalopathies)": [
                "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1",
                "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+",
                "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2"
            ],
            "🟢 Normal Brain Tissue": [
                "NORMAL T1",
                "NORMAL T2"
            ]
        }

        for group_name, classes in class_groups.items():
            st.markdown(f"#### {group_name}")
            for class_name in classes:
                color = CLASS_COLORS.get(class_name, "#6c757d")
                st.markdown(f"""
                <div class=\"class-item\">
                    <span style=\"color: {color}; font-size: 1.2em;\">●</span> {class_name}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### 📖 Clinical Resources
        
        - [American Brain Tumor Association](https://www.abta.org/) - Patient education and research
        - [National Brain Tumor Society](https://braintumor.org/) - Advocacy and clinical trials
        - [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) - Peer-reviewed research articles
        - [Radiopaedia](https://radiopaedia.org/) - Medical imaging reference database
        - [WHO Classification of CNS Tumors](https://www.who.int/publications/i/item/9789240058521) - Official tumor classification
        
        **Note:** This tool is for educational and research purposes. Always consult qualified medical professionals for diagnosis and treatment.
        """)
    
    def render_detection(self):
        """Render the detection page."""
        st.markdown("## 🔍 Neuropathology Detection")
        
        # Model loading section
        if not st.session_state.model_loaded:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ No Model Loaded</h4>
            <p>Please load a trained model to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.available_models:
                st.markdown("### 📦 Load Trained Model")
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_model = st.selectbox(
                        "Select a model to load:",
                        st.session_state.available_models,
                        help="Choose from available trained models in the models/ directory"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🚀 Load Model", use_container_width=True):
                        with st.spinner(f"Loading {selected_model}..."):
                            if self._load_model(selected_model):
                                st.success(f"✅ Model '{selected_model}' loaded successfully!")
                                st.rerun()
            else:
                st.error("No trained models found in the `models/` directory. Please train a model first using `train.py`.")
            return
        
        # Model loaded - show analysis interface with test accuracy and change model option
        col1, col2 = st.columns([2, 1])
        with col1:
            # Get test accuracy for display
            test_acc = "-"
            if 'metrics' in st.session_state and isinstance(st.session_state.metrics, dict):
                try:
                    test_acc = f"{float(st.session_state.metrics.get('accuracy', 0))*100:.2f}%"
                except Exception:
                    pass
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Model Loaded: {st.session_state.get('loaded_model_name', 'N/A')}</h4>
            <p><strong>📊 Test Accuracy:</strong> {test_acc} | Ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            # Change model dropdown - show current model as default
            current_model = st.session_state.get('loaded_model_name', '')
            # Create list with current model first, then others
            model_options = [current_model] + [m for m in st.session_state.available_models if m != current_model]
            
            change_model = st.selectbox(
                "🔄 Change Model:",
                model_options,
                key="change_model_selector",
                help="Select a different model to compare results"
            )
            
            if change_model != current_model:
                with st.spinner(f"Loading model: {change_model}..."):
                    if self._load_model(change_model):
                        # Clear previous analysis results when changing model
                        if 'analysis_results' in st.session_state:
                            del st.session_state.analysis_results
                        st.success(f"✅ Switched to: {change_model}")
                        st.rerun()
        
        st.markdown("""
        <div class="info-box">
        Upload a brain MRI image to detect potential pathologies. The system will analyze 
        the image using deep learning and provide predictions with confidence scores.
        </div>
        """, unsafe_allow_html=True)

        # Clear button if image exists
        if st.session_state.uploaded_image is not None:
            if st.button("🗑️ Clear Image & Results", use_container_width=False):
                st.session_state.uploaded_image = None
                st.session_state.image_source = None
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                st.rerun()
        
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

        # Check if we have a stored image
        if st.session_state.uploaded_image is not None and uploaded_file is None and not (image_url and fetch_from_url):
            image = st.session_state.uploaded_image
            src_label = st.session_state.image_source
        elif uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                src_label = 'Uploaded MRI Scan'
                # Store in session state
                st.session_state.uploaded_image = image
                st.session_state.image_source = src_label
            except Exception as e:
                st.error(f"Could not open uploaded image: {e}")
        elif image_url and fetch_from_url:
            try:
                req = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=10) as resp:
                    data = resp.read()
                image = Image.open(BytesIO(data))
                src_label = 'Fetched MRI Scan'
                # Store in session state
                st.session_state.uploaded_image = image
                st.session_state.image_source = src_label
            except Exception as e:
                st.error(f"Failed to fetch image from URL: {e}")

        if image is not None:
            # Display uploaded image and predicted diagnosis side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                # Show image at natural size within column width (no caption)
                st.image(image, use_container_width=True)
            
            with col2:
                # Add a bit more top padding before image details to improve alignment
                st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

                # Image details first to occupy top of right column
                mode_desc = {
                    'L': 'Grayscale (8-bit pixels)',
                    'RGB': 'True Color (3×8-bit pixels)',
                    'RGBA': 'True Color with transparency',
                    'P': 'Palette mode',
                    '1': 'Binary (1-bit pixels)'
                }.get(image.mode, image.mode)

                st.markdown(
                    f"""
                    <div class=\"info-box\"> 
                        <h4>🖼️ Image Details</h4>
                        <ul style=\"margin: 0; padding-left: 1.2rem;\"> 
                            <li><strong>Size:</strong> {image.size[0]}×{image.size[1]} pixels</li>
                            <li><strong>Mode:</strong> {mode_desc}</li>
                            <li><strong>Source:</strong> {src_label}</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Spacer between image details and analysis
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                st.markdown("### 🤖 Analysis")
                
                # Show analyze button if no results yet
                if 'analysis_results' not in st.session_state:
                    st.markdown("Click below to start the analysis with the loaded model.")
                    
                    # Analyze button
                    if st.button("🔬 Analyze Image", key="analyze", use_container_width=True):
                        with st.spinner("Analyzing image with deep learning model..."):
                            # Preprocess image
                            try:
                                # Save temp image for preprocessing
                                temp_path = "/tmp/uploaded_image.jpg"
                                image.convert('RGB').save(temp_path)
                                
                                # Load and preprocess
                                img_array = self.data_loader.load_and_preprocess_image(temp_path)
                                img_batch = np.expand_dims(img_array, axis=0)
                                
                                # Make prediction using trained model
                                if st.session_state.model_loaded and st.session_state.model is not None:
                                    predictions = st.session_state.model.predict(img_batch)
                                else:
                                    st.error("Model not loaded properly. Please reload the model.")
                                    return
                                
                                # Get predicted class and confidence
                                predicted_class = np.argmax(predictions[0])
                                confidence = predictions[0][predicted_class]
                                class_name = self.class_names[predicted_class]
                                
                                # Store in session state for display
                                st.session_state.analysis_results = {
                                    'predictions': predictions[0],
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'class_name': class_name
                                }
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error analyzing image: {str(e)}")
                else:
                    # Display predicted diagnosis in right column
                    results = st.session_state.analysis_results
                    class_color = CLASS_COLORS.get(results['class_name'], "#6c757d")
                    confidence = results['confidence']
                    class_name = results['class_name']
                    
                    # Determine confidence status
                    threshold = st.session_state.get('confidence_threshold', 0.7)
                    if confidence >= threshold:
                        status_icon = "✅"
                        status_text = "High Confidence"
                    elif confidence >= 0.5:
                        status_icon = "⚠️"
                        status_text = "Moderate Confidence"
                    else:
                        status_icon = "❌"
                        status_text = "Low Confidence"
                    
                    st.markdown(f"""
                    <div class="diagnosis-box" style="background: linear-gradient(135deg, {class_color}15, {class_color}05); border-left: 5px solid {class_color}; padding: 20px; border-radius: 10px;">
                    <h3>{status_icon} Predicted Diagnosis</h3>
                    <h2 style="color: {class_color};">
                        <span style="font-size: 0.8em;">●</span> {class_name}
                    </h2>
                    <p style="font-size: 1.2em;"><strong>Confidence:</strong> {confidence*100:.2f}%</p>
                    <p><strong>Status:</strong> {status_text} (threshold: {threshold*100:.0f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Button to re-analyze
                    if st.button("🔄 Re-analyze", key="reanalyze", use_container_width=True):
                        del st.session_state.analysis_results
                        st.rerun()

                # End of right column content
            
            # Display detailed results in full-width section if available
            if 'analysis_results' in st.session_state:
                st.markdown("---")
                results = st.session_state.analysis_results
                self.display_detailed_results(
                    results['predictions'],
                    results['predicted_class'],
                    results['class_name']
                )
        
        else:
            st.info("👆 Please load a model and upload a brain MRI image to begin analysis")
    
    def display_detailed_results(self, predictions: np.ndarray, predicted_class: int, class_name: str):
        
        # Get color for the predicted class
        class_color = CLASS_COLORS.get(class_name, "#6c757d")
        
        # Detailed results - Full width
        st.markdown("### 📊 Detailed Analysis")
        
        # Create short labels for plotting (remove parentheses content)
        short_labels = []
        for name in self.class_names:
            # Remove content in parentheses and trim
            if '(' in name:
                short_name = name.split('(')[0].strip() + ' ' + name.split()[-1]
            else:
                short_name = name
            short_labels.append(short_name)
        
        # Confidence scores for all classes - responsive visualization with hover darkening
        # All bars use their assigned color codes (not greyed out)
        import colorsys
        
        bar_colors = []
        hover_label_colors = []
        
        for i, name in enumerate(self.class_names):
            # Use the assigned color for each class
            base_color = CLASS_COLORS.get(name, '#6c757d')
            bar_colors.append(base_color)
            
            # Darken the color for hover tooltip by reducing lightness
            hex_color = base_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            # Reduce lightness by 30% for hover
            darker_l = max(0, l * 0.7)
            r_dark, g_dark, b_dark = colorsys.hls_to_rgb(h, darker_l, s)
            darker_hex = '#{:02x}{:02x}{:02x}'.format(
                int(r_dark * 255), int(g_dark * 255), int(b_dark * 255)
            )
            hover_label_colors.append(darker_hex)
        
        # Create bar chart with individual traces for custom hover tooltip colors
        fig = go.Figure()
        
        for i, (label, pred, color, hover_label_color) in enumerate(
            zip(short_labels, predictions, bar_colors, hover_label_colors)
        ):
            # Make predicted class bar slightly more opaque
            opacity = 1.0 if i == predicted_class else 0.85
            
            fig.add_trace(go.Bar(
                x=[label],
                y=[pred],
                marker=dict(
                    color=color,
                    opacity=opacity,
                    line=dict(width=0)
                ),
                text=f'{pred*100:.1f}%',
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>',
                hoverlabel=dict(
                    bgcolor=hover_label_color,
                    font_size=13,
                    font_family="sans-serif",
                    font_color="white",
                    bordercolor=hover_label_color
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Prediction Confidence for All Classes',
            xaxis_title='Pathology Type',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1], tickformat='.0%', showspikes=False),
            xaxis=dict(tickangle=-45, showspikes=False),
            height=500,
            hovermode='closest',
            bargap=0.15
        )
        
        st.plotly_chart(fig, use_container_width=True, key="confidence_chart")
        
        
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
        
        # All class probabilities - sorted and styled
        st.markdown("### 📋 All Class Probabilities")
        
        # Create sorted list of (name, probability, color) tuples
        prob_data = [(name, prob, CLASS_COLORS.get(name, "#6c757d")) 
                     for name, prob in zip(self.class_names, predictions)]
        prob_data_sorted = sorted(prob_data, key=lambda x: x[1], reverse=True)
        
        # Display top 5
        st.markdown("**Top 5 Predictions:**")
        for i, (name, prob, color) in enumerate(prob_data_sorted[:5], 1):
            st.markdown(f"""
            <div style="margin-bottom: 12px; padding: 10px; background: linear-gradient(90deg, {color}20 0%, {color}05 100%); border-radius: 8px; border-left: 4px solid {color};">
                <span style="font-weight: 600; color: {color};">#{i}</span> 
                <span style="color: {color}; font-size: 1.1em;">●</span> 
                <strong>{name}</strong>: {prob*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))
        
        # Expandable section for remaining probabilities
        if len(prob_data_sorted) > 5:
            with st.expander(f"📊 View all {len(prob_data_sorted)} class probabilities"):
                for i, (name, prob, color) in enumerate(prob_data_sorted[5:], 6):
                    st.markdown(f"""
                    <div style="margin-bottom: 12px; padding: 10px; background: linear-gradient(90deg, {color}20 0%, {color}05 100%); border-radius: 8px; border-left: 4px solid {color};">
                        <span style="font-weight: 600; color: {color};">#{i}</span> 
                        <span style="color: {color}; font-size: 1.1em;">●</span> 
                        <strong>{name}</strong>: {prob*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(prob))
    
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
            8. Output Layer (17 units, Softmax activation)
            
            **Total Parameters:** 3.05 Million
            - Trainable: 0.79 Million
            - Frozen (base): 2.26 Million
            
            ### 🖥️ Trained On
            
            **Hardware:**
            - GPU: NVIDIA® GeForce MX330
            - VRAM: 2GB
            
            **Software Environment:**
            - CUDA® Toolkit: 12.3
            - cuDNN SDK: 8.9.7
            - NVIDIA® GPU Driver: Version 576.83
            - WSL Version: 2.6.1.0
            - OS Image: Ubuntu 24.04.1 LTS
            - Host OS: Windows 10.0.26200.6899
            """)
        
        with col2:
            st.markdown("""
            ### 📚 Training Strategy
            
            **Two-Stage Training:**
            
            **Stage 1: Feature Learning**
            - Freeze base model weights
            - Train only classification head
            - Learn task-specific features
            - Epochs: 50
            - Learning rate: 0.001
            
            **Stage 2: Fine-Tuning**
            - Unfreeze last 20 layers
            - Fine-tune with lower learning rate
            - Adapt features to medical images
            - Epochs: 70
            - Learning rate: 0.0005
            
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
        
        # Model performance (dynamic based on loaded metrics)
        st.markdown("### 📈 Performance (Test Set)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Extract metrics from loaded model or show defaults
        accuracy = 0.7813
        precision = 0.8233
        recall = 0.7813
        f1_score = 0.7744
        
        if 'metrics' in st.session_state and isinstance(st.session_state.metrics, dict):
            try:
                accuracy = float(st.session_state.metrics.get('accuracy', accuracy))
                if 'classification_report' in st.session_state.metrics:
                    report = st.session_state.metrics['classification_report']
                    if 'weighted avg' in report:
                        precision = report['weighted avg'].get('precision', precision)
                        recall = report['weighted avg'].get('recall', recall)
                        f1_score = report['weighted avg'].get('f1-score', f1_score)
            except Exception:
                pass
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h2>{accuracy*100:.2f}%</h2>
            <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h2>{precision*100:.2f}%</h2>
            <p>Precision (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <h2>{recall*100:.2f}%</h2>
            <p>Recall (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
            <h2>{f1_score*100:.2f}%</h2>
            <p>F1-Score (weighted)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")


def main():
    """Main entry point for the application."""
    app = NeuropathologyApp()
    app.run()


if __name__ == "__main__":
    main()
