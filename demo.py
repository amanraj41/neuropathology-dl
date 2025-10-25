"""
Demo Script for Neuropathology Detection System

This script demonstrates the functionality of the system without requiring
a full dataset. It creates synthetic data and tests all major components.

Usage:
    python demo.py
"""

import sys
import os
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.neuropathology_model import NeuropathologyModel
from src.data.data_loader import MRIDataLoader, DataGenerator
from src.utils.helpers import (
    Visualizer, ModelEvaluator, get_class_names,
    get_class_descriptions, create_sample_data
)


def test_data_loader():
    """Test the data loader functionality."""
    print("\n" + "="*70)
    print("Testing Data Loader")
    print("="*70)
    
    # Create a synthetic image
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    temp_path = "/tmp/test_image.jpg"
    img.save(temp_path)
    
    # Test loading
    loader = MRIDataLoader(img_size=(224, 224))
    loaded_img = loader.load_and_preprocess_image(temp_path)
    
    print(f"✓ Image loaded successfully")
    print(f"  Original size: 300×300")
    print(f"  Processed size: {loaded_img.shape}")
    print(f"  Data range: [{loaded_img.min():.3f}, {loaded_img.max():.3f}]")
    
    # Test augmentation
    augmented = loader.augment_image(loaded_img)
    print(f"✓ Data augmentation working")
    
    # Clean up
    os.remove(temp_path)
    
    return True


def test_model_building():
    """Test model building and architecture."""
    print("\n" + "="*70)
    print("Testing Model Building")
    print("="*70)
    
    # Test each architecture
    architectures = ['efficientnet', 'resnet', 'vgg', 'mobilenet']
    
    for arch in architectures:
        print(f"\nTesting {arch}...")
        try:
            model_wrapper = NeuropathologyModel(
                num_classes=4,
                input_shape=(224, 224, 3),
                base_model=arch,
                trainable_layers=20
            )
            
            model = model_wrapper.build_model()
            model_wrapper.compile_model(learning_rate=0.001)
            
            # Get total parameters
            total_params = model.count_params()
            
            print(f"✓ {arch} model built successfully")
            print(f"  Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"✗ Error building {arch}: {str(e)}")
            return False
    
    return True


def test_prediction():
    """Test model prediction functionality."""
    print("\n" + "="*70)
    print("Testing Model Prediction")
    print("="*70)
    
    # Build a simple model
    model_wrapper = NeuropathologyModel(
        num_classes=4,
        input_shape=(224, 224, 3),
        base_model='mobilenet',  # Fastest for demo
        trainable_layers=5
    )
    
    model = model_wrapper.build_model()
    model_wrapper.compile_model()
    
    # Create synthetic batch
    batch_size = 4
    images = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    
    # Make predictions
    predictions = model_wrapper.predict(images)
    predicted_classes, confidences = model_wrapper.predict_class(images)
    
    print(f"✓ Predictions successful")
    print(f"  Batch size: {batch_size}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predicted classes: {predicted_classes}")
    print(f"  Confidences: {confidences}")
    
    # Verify predictions sum to 1
    sums = predictions.sum(axis=1)
    assert np.allclose(sums, 1.0), "Predictions don't sum to 1!"
    print(f"✓ Softmax output verified (probabilities sum to 1)")
    
    return True


def test_data_generator():
    """Test the data generator."""
    print("\n" + "="*70)
    print("Testing Data Generator")
    print("="*70)
    
    # Create synthetic data
    num_samples = 100
    images, labels = create_sample_data(num_samples=num_samples, num_classes=4)
    
    # Create temporary image files
    temp_dir = "/tmp/demo_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    for i in range(num_samples):
        path = os.path.join(temp_dir, f"image_{i}.jpg")
        img = Image.fromarray((images[i] * 255).astype(np.uint8))
        img.save(path)
        image_paths.append(path)
    
    # Create generator
    generator = DataGenerator(
        image_paths, labels,
        batch_size=16,
        augment=True,
        shuffle=True
    )
    
    print(f"✓ Generator created")
    print(f"  Total samples: {num_samples}")
    print(f"  Batches per epoch: {len(generator)}")
    print(f"  Batch size: 16")
    
    # Test getting a batch
    batch_x, batch_y = generator[0]
    print(f"✓ Batch retrieved")
    print(f"  Batch images shape: {batch_x.shape}")
    print(f"  Batch labels shape: {batch_y.shape}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return True


def test_visualization():
    """Test visualization utilities."""
    print("\n" + "="*70)
    print("Testing Visualization")
    print("="*70)
    
    # Create synthetic training history
    history = {
        'accuracy': list(np.linspace(0.6, 0.95, 30)),
        'val_accuracy': list(np.linspace(0.55, 0.93, 30)),
        'loss': list(np.linspace(1.2, 0.2, 30)),
        'val_loss': list(np.linspace(1.3, 0.25, 30)),
        'precision': list(np.linspace(0.6, 0.94, 30)),
        'val_precision': list(np.linspace(0.58, 0.92, 30)),
        'recall': list(np.linspace(0.58, 0.93, 30)),
        'val_recall': list(np.linspace(0.56, 0.91, 30)),
    }
    
    visualizer = Visualizer()
    
    try:
        # Test training history plot
        fig = visualizer.plot_training_history(history)
        print("✓ Training history plot created")
        
        # Test confusion matrix
        y_true = np.random.randint(0, 4, 100)
        y_pred = np.random.randint(0, 4, 100)
        class_names = get_class_names()
        
        cm_fig = visualizer.plot_confusion_matrix(y_true, y_pred, class_names)
        print("✓ Confusion matrix plot created")
        
        # Test prediction confidence plot
        predictions = np.random.dirichlet(np.ones(4) * 2, size=1)[0]
        conf_fig = visualizer.plot_prediction_confidence(predictions, class_names)
        print("✓ Prediction confidence plot created")
        
    except Exception as e:
        print(f"✗ Visualization error: {str(e)}")
        return False
    
    return True


def test_utilities():
    """Test utility functions."""
    print("\n" + "="*70)
    print("Testing Utility Functions")
    print("="*70)
    
    # Test class names
    class_names = get_class_names()
    print(f"✓ Class names: {class_names}")
    
    # Test class descriptions
    descriptions = get_class_descriptions()
    print(f"✓ Class descriptions loaded: {len(descriptions)} classes")
    
    # Test sample data creation
    images, labels = create_sample_data(num_samples=50, num_classes=4)
    print(f"✓ Sample data created")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return True


def run_all_tests():
    """Run all demo tests."""
    print("\n" + "="*70)
    print("NEUROPATHOLOGY DETECTION SYSTEM - DEMO")
    print("="*70)
    print("\nThis demo tests all major components of the system.")
    print("It does not require a real dataset or trained model.")
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model Building", test_model_building),
        ("Model Prediction", test_prediction),
        ("Data Generator", test_data_generator),
        ("Visualization", test_visualization),
        ("Utilities", test_utilities),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe system is working correctly.")
        print("\nNext steps:")
        print("1. Obtain a brain MRI dataset")
        print("2. Train the model: python train.py --data_dir /path/to/data")
        print("3. Run the web app: streamlit run app.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above and ensure all dependencies are installed.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
