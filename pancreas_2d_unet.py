"""
2D PANCREAS SEGMENTATION - PNG Images with Data Generators (Memory Efficient)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = './data'
IMAGES_DIR = Path(DATA_PATH) / 'images'
LABELS_DIR = Path(DATA_PATH) / 'labels'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# 2D U-NET MODEL
# ============================================================================

def create_2d_unet(input_shape=(192, 192, 1)):
    """2D U-Net for image segmentation"""
    
    print("\n🧠 Building 2D U-Net...")
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    
    # Decoder
    u4 = layers.UpSampling2D(2)(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    print("✓ Model built successfully")
    
    return model

# ============================================================================
# DATA GENERATOR
# ============================================================================

class PancreasDataGenerator(keras.utils.Sequence):
    """Custom data generator for memory efficiency"""
    
    def __init__(self, image_paths, mask_paths, batch_size=16, img_size=(192, 192)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.indices = np.arange(len(self.image_paths))
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            # Load image
            img = cv2.imread(str(self.image_paths[i]), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            
            # Load mask
            mask = cv2.imread(str(self.mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            mask = mask.astype(np.float32) / 255.0
            
            X_batch.append(img[..., np.newaxis])
            y_batch.append(mask[..., np.newaxis])
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ============================================================================
# PREPARE FILE LISTS
# ============================================================================

def get_file_lists():
    """Get lists of image and mask files"""
    
    image_files = []
    mask_files = []
    
    # Get subdirectories
    subdirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    
    print(f"\n📂 Found {len(subdirs)} subdirectories")
    print("=" * 60)
    
    total_pairs = 0
    
    for subdir in subdirs:
        subdir_name = subdir.name
        images = sorted(subdir.glob('*.png'))
        masks = sorted((LABELS_DIR / subdir_name).glob('*.png'))
        
        num_pairs = min(len(images), len(masks))
        
        print(f"📁 {subdir_name}: {num_pairs} pairs")
        
        image_files.extend(images[:num_pairs])
        mask_files.extend(masks[:num_pairs])
        total_pairs += num_pairs
    
    print("=" * 60)
    print(f"✓ Total: {total_pairs} image-mask pairs")
    
    return np.array(image_files), np.array(mask_files)

# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_files, y_files, epochs=20, batch_size=16):
    """Train using data generators"""
    
    # Split files
    X_train, X_test, y_train, y_test = train_test_split(
        X_files, y_files, test_size=0.2, random_state=42
    )
    
    print(f"\n🔧 TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image size: 192x192")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    # Create generators
    train_gen = PancreasDataGenerator(X_train, y_train, batch_size=batch_size)
    test_gen = PancreasDataGenerator(X_test, y_test, batch_size=batch_size)
    
    # Create model
    model = create_2d_unet()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🚀 Starting training...\n")
    
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        verbose=1
    )
    
    model_path = os.path.join(MODELS_DIR, 'pancreas_2d_unet.h5')
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    return model, history, (X_test, y_test)

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(image_path, mask_path, model):
    """Visualize results"""
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (192, 192))
    img_norm = img.astype(np.float32) / 255.0
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (192, 192))
    mask_norm = mask.astype(np.float32) / 255.0
    
    # Predict
    pred = model.predict(img_norm[np.newaxis, ..., np.newaxis], verbose=0)
    pred = pred[0, :, :, 0]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_norm, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(img_norm, cmap='gray')
    axes[2].imshow(pred, cmap='Reds', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("🏥 2D PANCREAS SEGMENTATION - PNG Images")
    print("="*60)
    
    print("\n📂 Preparing dataset...")
    X_files, y_files = get_file_lists()
    
    if len(X_files) > 0:
        model, history, test_data = train_model(X_files, y_files, epochs=20, batch_size=16)
        
        if model:
            print("\n📊 Testing...")
            X_test, y_test = test_data
            visualize_results(X_test[0], y_test[0], model)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {MODELS_DIR}/pancreas_2d_unet.h5")