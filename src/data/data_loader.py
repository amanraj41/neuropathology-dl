"""
Data Loading and Preprocessing for Neuropathology Detection

This module provides utilities for loading, preprocessing, and augmenting
brain MRI images for deep learning model training and inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from typing import Tuple, List
import os


class MRIDataLoader:
    """
    MRI image loader and preprocessor.
    
    Handles loading and preprocessing of brain MRI images for model input.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the data loader.
        
        Args:
            img_size: Target image size (height, width)
        """
        self.img_size = img_size
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image array of shape (height, width, 3)
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(self.img_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
    
    def load_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess a batch of images.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            Batch of preprocessed images of shape (batch_size, height, width, 3)
        """
        images = []
        for path in image_paths:
            img = self.load_and_preprocess_image(path)
            images.append(img)
        
        return np.array(images)


class DataGenerator(keras.utils.Sequence):
    """
    Keras data generator for training and evaluation.
    
    Generates batches of images with optional data augmentation.
    """
    
    def __init__(self,
                 image_paths: List[str],
                 labels: np.ndarray,
                 batch_size: int = 32,
                 img_size: Tuple[int, int] = (224, 224),
                 augment: bool = False,
                 shuffle: bool = True):
        """
        Initialize the data generator.
        
        Args:
            image_paths: List of paths to image files
            labels: Array of labels (one-hot encoded)
            batch_size: Number of samples per batch
            img_size: Target image size
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data after each epoch
        """
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        # Data augmentation layers
        if self.augment:
            self.augmentation = keras.Sequential([
                keras.layers.RandomRotation(0.2),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomFlip("horizontal"),
            ])
        
        self.on_epoch_end()
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (images, labels) for the batch
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Get batch data
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]
        
        # Load and preprocess images
        batch_images = self._load_batch(batch_paths)
        
        # Apply augmentation if enabled
        if self.augment:
            batch_images = self.augmentation(batch_images, training=True)
        
        return batch_images, batch_labels
    
    def _load_batch(self, paths: List[str]) -> np.ndarray:
        """
        Load a batch of images.
        
        Args:
            paths: List of image paths
        
        Returns:
            Batch of preprocessed images
        """
        images = []
        for path in paths:
            try:
                # Load image
                img = Image.open(path)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize(self.img_size, Image.LANCZOS)
                
                # Convert to array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create a black image as fallback
                images.append(np.zeros((*self.img_size, 3), dtype=np.float32))
        
        return np.array(images)
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch if shuffle is True."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
