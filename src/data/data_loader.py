"""
Data loading and preprocessing for brain MRI images.

This module includes:
- MRIDataLoader: utility for loading and preprocessing single images
- DataGenerator: a tf.keras.utils.Sequence for batched loading with augmentation

Design goals:
- Minimal external dependencies (PIL + TensorFlow ops)
- Deterministic shapes and dtypes
- Safe defaults for normalization and augmentation
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf


class MRIDataLoader:
    """
    Helper to load, preprocess, and augment MRI images.

    Contract:
    - Inputs: file path (str) or HxWxC numpy array in [0, 255] or [0, 1]
    - Outputs: float32 numpy array of shape (H, W, 3) in [0, 1]
    - img_size: (height, width) target size
    """

    def __init__(self, img_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        self.img_size = img_size
        self.normalize = normalize

    def load_image(self, filepath: str) -> np.ndarray:
        """Load an image from disk as RGB uint8 array.

        Args:
            filepath: Path to image file

        Returns:
            HxWx3 uint8 numpy array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")

        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.resize(self.img_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
        return arr

    def preprocess_array(self, img_array: np.ndarray) -> np.ndarray:
        """Resize (if needed) and normalize to float32 [0, 1].

        Args:
            img_array: HxWxC array in uint8 [0,255] or float [0,1]

        Returns:
            float32 array HxWx3 in [0,1]
        """
        if img_array.ndim != 3:
            raise ValueError(f"Expected 3D array (H,W,C), got shape {img_array.shape}")

        # Ensure 3 channels
        if img_array.shape[2] == 4:
            # Drop alpha channel
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)

        # Resize with TF to be consistent
        tensor = tf.convert_to_tensor(img_array)
        # Ensure type float32 and resize first
        tensor = tf.image.resize(tensor, self.img_size, method=tf.image.ResizeMethod.BILINEAR)
        tensor = tf.cast(tensor, tf.float32)

        # Normalize to [0,1] if requested
        if self.normalize:
            # If values look like 0..255, scale down
            maxv = tf.reduce_max(tensor)
            tensor = tf.cond(maxv > 1.5, lambda: tensor / 255.0, lambda: tensor)
        else:
            # Keep 0..255 range
            pass

        return tensor.numpy()

    def load_and_preprocess_image(self, filepath: str) -> np.ndarray:
        """Convenience: load from file and preprocess.

        Returns:
            float32 array HxWx3 in [0,1]
        """
        arr = self.load_image(filepath)
        return self.preprocess_array(arr)

    def augment_image(self, img_array: np.ndarray) -> np.ndarray:
        """Apply light augmentation suitable for MRI classification.

        Note: expects float32 array in [0,1]. Returns same shape/dtype.
        """
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        tensor = tf.convert_to_tensor(img_array)

        # Random horizontal flip
        tensor = tf.image.random_flip_left_right(tensor)
        
        # Random vertical flip (medical images have anatomical symmetry)
        tensor = tf.image.random_flip_up_down(tensor)

        # Random rotation (-25° .. +25°) - increased from 20°
        rot = tf.keras.layers.RandomRotation(
            factor=25.0/360.0, fill_mode='reflect', interpolation='bilinear'
        )
        tensor = rot(tf.expand_dims(tensor, 0), training=True)[0]

        # Random zoom via central crop (75% to 100%) - increased range from 90%
        zoom_factor = tf.random.uniform([], 0.75, 1.0)
        crop_h = tf.cast(tf.round(self.img_size[0] * zoom_factor), tf.int32)
        crop_w = tf.cast(tf.round(self.img_size[1] * zoom_factor), tf.int32)
        tensor = tf.image.resize_with_crop_or_pad(tensor, crop_h, crop_w)
        tensor = tf.image.resize(tensor, self.img_size, method=tf.image.ResizeMethod.BILINEAR)
        
        # Random brightness adjustment (simulate different scanner intensities)
        tensor = tf.image.random_brightness(tensor, max_delta=0.2)
        
        # Random contrast adjustment
        tensor = tf.image.random_contrast(tensor, lower=0.8, upper=1.2)

        return tf.clip_by_value(tensor, 0.0, 1.0).numpy()


        


class DataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for loading images from file paths.

    Features:
    - On-the-fly loading and preprocessing
    - Optional augmentation for training
    - Shuffling at epoch end
    - Works with one-hot labels (preferred)

    Inputs:
    - image_paths: list of file paths
    - labels: np.ndarray of shape (N, num_classes) one-hot OR shape (N,) class indices
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        batch_size: int = 32,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        shuffle: bool = True,
        loader: Optional[MRIDataLoader] = None,
    ):
        self.image_paths = list(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.loader = loader or MRIDataLoader(img_size=img_size)

        # Ensure labels are two-dimensional (one-hot)
        if self.labels.ndim == 1:
            # Convert class indices to one-hot
            num_classes = int(self.labels.max()) + 1
            labels_onehot = np.zeros((len(self.labels), num_classes), dtype=np.float32)
            labels_onehot[np.arange(len(self.labels)), self.labels.astype(int)] = 1.0
            self.labels = labels_onehot
        elif self.labels.ndim == 2:
            self.labels = self.labels.astype(np.float32)
        else:
            raise ValueError(f"Labels must be 1D or 2D, got shape {self.labels.shape}")

        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_ids = self.indices[start:end]

        batch_paths = [self.image_paths[i] for i in batch_ids]
        batch_labels = self.labels[batch_ids]

        batch_images = []
        for p in batch_paths:
            img = self.loader.load_and_preprocess_image(p)
            if self.augment:
                img = self.loader.augment_image(img)
            batch_images.append(img)

        batch_x = np.stack(batch_images, axis=0).astype(np.float32)
        batch_y = batch_labels.astype(np.float32)
        return batch_x, batch_y
