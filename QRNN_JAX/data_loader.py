import os
import random
import numpy as np
import tensorflow as tf
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import sobel
from sklearn.linear_model import LogisticRegression

# Ensure reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_digits_data(n_train, n_test=100, encoding_type='angle'):
    """Load and preprocess Digits dataset."""
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    if n_train > len(X) * 0.8 or n_test > len(X) * 0.2:
        raise ValueError(f"Requested n_train={n_train} or n_test={n_test} exceeds available samples ({len(X)} total).")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]
    
    X_train, X_test = X_train.reshape(-1, 8, 8), X_test.reshape(-1, 8, 8)
    X_train = torch.tensor(X_train).unsqueeze(1).float()
    X_test = torch.tensor(X_test).unsqueeze(1).float()

    pooled_train = F.adaptive_avg_pool2d(X_train, output_size=(3, 3))
    pooled_test = F.adaptive_avg_pool2d(X_test, output_size=(3, 3))

    train_seq = pooled_train.squeeze(1)
    test_seq = pooled_test.squeeze(1)

    _min, _max = train_seq.min(), train_seq.max()
    train_seq = ((train_seq - _min) / (_max - _min)) * torch.pi
    test_seq = ((test_seq - _min) / (_max - _min)) * torch.pi

    train = train_seq.reshape(-1, train_seq.shape[1] * train_seq.shape[2])
    test = test_seq.reshape(-1, test_seq.shape[1] * test_seq.shape[2])
    
    if encoding_type == 'angle':
        return (
            jnp.array(train.numpy(), dtype=jnp.float32),
            jnp.array(test.numpy(), dtype=jnp.float32),
            jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1),
            jnp.array(y_test, dtype=jnp.float32).reshape(-1, 1)
        )
    else:
        raise ValueError('Unknown encoding type')

def build_cnn_classifier(input_shape=(32, 32, 3), output_dim=4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    features = layers.Dense(output_dim, activation='tanh', name='feature_layer')(x)
    outputs = layers.Dense(10, activation='softmax')(features)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_and_save_cnn(x_train, y_train, output_dim=4, epochs=5, save_path='cnn_model_cifar.h5'):
    """Train and save CNN, returning validation accuracy."""
    print(f"Training CNN for {epochs} epochs...")
    model = build_cnn_classifier(input_shape=x_train.shape[1:], output_dim=output_dim)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1, 
        shuffle=True
    )
    model.save(save_path)
    print(f"CNN saved to {save_path}")
    # Return the final validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]
    return model, val_accuracy

def load_cnn_feature_model(save_path='cnn_model_cifar.h5', output_dim=4):
    full_model = tf.keras.models.load_model(save_path)
    feature_model = models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer('feature_layer').output
    )
    feature_model.trainable = False
    return feature_model

def evaluate_cnn_model(model, x_val, y_val):
    """Evaluate CNN model accuracy on validation data."""
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy

def test_cnn_features(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.array(X_train), np.array(y_train))
    return clf.score(np.array(X_train), np.array(y_train))

def load_cifar2(n_train=100, n_test=100, seq_len=4, classify_choice=[0, 1], train_cnn_epochs=None, force_retrain=False):
    """Load and preprocess CIFAR-10 with CNN feature extraction and automatic retraining."""
    if train_cnn_epochs is None:
        train_cnn_epochs = 100

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Filter for the two chosen classes
    train_mask = (y_train == classify_choice[0]) | (y_train == classify_choice[1])
    test_mask = (y_test == classify_choice[0]) | (y_test == classify_choice[1])
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Validate sample sizes
    if n_train > len(x_train) or n_test > len(x_test):
        print(f'Requested n_train={n_train} or n_test={n_test} exceeds available samples (train={len(x_train)}, test={len(x_test)}).')
        print(f'Setting n_train={len(x_train)} and n_test={len(x_test)}.')
        n_train = len(x_train)
        n_test = len(x_test)

    # Map labels: classify_choice[0] -> 0, classify_choice[1] -> 1
    y_train = np.where(y_train == classify_choice[0], 0, 1).astype(np.float32)
    y_test = np.where(y_test == classify_choice[0], 0, 1).astype(np.float32)

    # Sample the requested number of examples
    np.random.seed(42)
    train_indices = np.random.choice(len(x_train), size=min(n_train, len(x_train)), replace=False)
    test_indices = np.random.choice(len(x_test), size=min(n_test, len(x_test)), replace=False)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Create validation set for evaluation
    val_size = int(0.1 * n_train)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Train or load CNN and extract features
    cnn_model_path = f'cnn_model_cifar_{n_train}train_{n_test}test_{train_cnn_epochs}_{classify_choice[0]}_{classify_choice[1]}.h5'
    max_retrain_attempts = 3
    attempt = 0
    val_accuracy = 0.0

    while attempt < max_retrain_attempts:
        if force_retrain or not os.path.exists(cnn_model_path) or val_accuracy < 0.8:
            print(f"Training CNN (attempt {attempt + 1}/{max_retrain_attempts})...")
            model, val_accuracy = train_and_save_cnn(
                x_train, y_train, 
                output_dim=seq_len, 
                epochs=train_cnn_epochs, 
                save_path=cnn_model_path
            )
            # Re-evaluate on validation set to confirm
            val_accuracy = evaluate_cnn_model(model, x_val, y_val)
            print(f"Validation accuracy: {val_accuracy:.3f}")
            if val_accuracy >= 0.8:
                print("CNN achieved sufficient accuracy. Proceeding...")
                break
            attempt += 1
            force_retrain = True  # Force retrain on next attempt
        else:
            print(f"Loading existing CNN model from {cnn_model_path}...")
            model = tf.keras.models.load_model(cnn_model_path)
            val_accuracy = evaluate_cnn_model(model, x_val, y_val)
            print(f"Validation accuracy of loaded model: {val_accuracy:.3f}")
            if val_accuracy >= 0.8:
                print("Loaded CNN model meets accuracy threshold. Proceeding...")
                break
            attempt += 1
            force_retrain = True  # Force retrain on next attempt

    if attempt == max_retrain_attempts and val_accuracy < 0.8:
        print(f"Warning: Max retrain attempts ({max_retrain_attempts}) reached. Proceeding with best model (val_accuracy={val_accuracy:.3f}).")

    # Extract features
    feature_model = load_cnn_feature_model(cnn_model_path, output_dim=seq_len)
    X_train_feat = feature_model.predict(x_train, verbose=0)
    X_test_feat = feature_model.predict(x_test, verbose=0)

    print(f"Logistic Regression accuracy on CNN features: {test_cnn_features(X_train_feat, y_train):.3f}")

    # Normalize features for angle encoding
    X_train_norm = np.linalg.norm(X_train_feat, axis=1, keepdims=True)
    X_test_norm = np.linalg.norm(X_test_feat, axis=1, keepdims=True)
    X_train_feat = X_train_feat / np.where(X_train_norm > 0, X_train_norm, 1.0)
    X_test_feat = X_test_feat / np.where(X_test_norm > 0, X_test_norm, 1.0)

    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train_feat, dtype=jnp.float32)
    X_test_jax = jnp.array(X_test_feat, dtype=jnp.float32)
    y_train_jax = jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1)
    y_test_jax = jnp.array(y_test, dtype=jnp.float32).reshape(-1, 1)

    return X_train_jax, y_train_jax, X_test_jax, y_test_jax

def load_mnist(n_train=100, n_test=100, patch_size=10, reduce_fn=np.max, classify_choice=[0, 1], show_size=False):
    """Load and preprocess MNIST with patch-based sequences."""
    def patchify_images(images):
        H, W = images.shape[1:]
        sequences = []
        for img in images:
            seq = [
                reduce_fn(img[i:i+patch_size, j:j+patch_size])
                for i in range(0, H, patch_size)
                for j in range(0, W, patch_size)
            ]
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)

    def normalize_sequence(seq, min_val=0.0, max_val=255.0):
        return (seq - min_val) / (max_val - min_val) * np.pi

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_mask = (y_train == classify_choice[0]) | (y_train == classify_choice[1])
    test_mask = (y_test == classify_choice[0]) | (y_test == classify_choice[1])
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if n_train > len(X_train) or n_test > len(X_test):
        print(f'Requested n_train={n_train} or n_test={n_test} exceeds available samples (train={len(X_train)}, test={len(X_test)}).')
        print(f'Setting n_train={len(X_train)} and n_test={len(X_test)}.')
        n_train, n_test = len(X_train), len(X_test)
    if 28 % patch_size != 0:
        print(f"Warning: patch_size={patch_size} does not evenly divide 28; some data may be cropped.")

    if show_size:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

    y_train = np.where(y_train == classify_choice[0], 0, 1).astype(np.float32)
    y_test = np.where(y_test == classify_choice[0], 0, 1).astype(np.float32)

    np.random.seed(42)
    train_indices = np.random.choice(len(X_train), size=min(n_train, len(X_train)), replace=False)
    test_indices = np.random.choice(len(X_test), size=min(n_test, len(X_test)), replace=False)

    X_train_sampled = X_train[train_indices]
    y_train_sampled = y_train[train_indices]
    X_test_sampled = X_test[test_indices]
    y_test_sampled = y_test[test_indices]

    X_train_seq = jnp.array(normalize_sequence(patchify_images(X_train_sampled)), dtype=jnp.float32)
    X_test_seq = jnp.array(normalize_sequence(patchify_images(X_test_sampled)), dtype=jnp.float32)
    y_train_bin = jnp.array(y_train_sampled.reshape(-1, 1), dtype=jnp.float32)
    y_test_bin = jnp.array(y_test_sampled.reshape(-1, 1), dtype=jnp.float32)

    return X_train_seq, y_train_bin, X_test_seq, y_test_bin

def load_old_cifar2(n_train=100, n_test=100, seq_len=4, reduce_fn=np.mean, classify_choice=[0, 1]):
    """Load and preprocess CIFAR-10 with Sobel edge detection and patch-based sequences."""
    def rgb2gray(images):
        return 0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2]

    def sobel_edges(images):
        edge_images = []
        for img in images:
            dx = sobel(img, axis=0, mode='constant')
            dy = sobel(img, axis=1, mode='constant')
            mag = np.hypot(dx, dy)
            edge_images.append(mag)
        return np.array(edge_images)

    def patchify_images(images):
        H, W = images.shape[1:]
        patch_size = H // seq_len
        sequences = []
        for img in images:
            seq = [
                reduce_fn(img[i:i+patch_size, j:j+patch_size])
                for i in range(0, H, patch_size)
                for j in range(0, W, patch_size)
            ]
            sequences.append(seq)
        return np.array(sequences)

    def normalize_sequence(seq, min_val=0.0, max_val=255.0):
        return (seq - min_val) / (max_val - min_val) * np.pi

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    train_mask = (y_train == classify_choice[0]) | (y_train == classify_choice[1])
    test_mask = (y_test == classify_choice[0]) | (y_test == classify_choice[1])
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    if n_train > len(x_train) or n_test > len(x_test):
        raise ValueError(f"Requested n_train={n_train} or n_test={n_test} exceeds available samples (train={len(x_train)}, test={len(x_test)}).")
    if 32 % seq_len != 0:
        print(f"Warning: seq_len={seq_len} does not evenly divide 32; some data may be cropped.")

    np.random.seed(42)
    train_indices = np.random.choice(len(x_train), size=min(n_train, len(x_train)), replace=False)
    test_indices = np.random.choice(len(x_test), size=min(n_test, len(x_test)), replace=False)

    x_train_filtered = x_train[train_indices]
    y_train_filtered = np.where(y_train[train_indices] == classify_choice[0], 0, 1).astype(np.float32)
    x_test_filtered = x_test[test_indices]
    y_test_filtered = np.where(y_test[test_indices] == classify_choice[0], 0, 1).astype(np.float32)

    X_train_gray = rgb2gray(x_train_filtered)
    X_test_gray = rgb2gray(x_test_filtered)

    X_train_edges = sobel_edges(X_train_gray)
    X_test_edges = sobel_edges(X_test_gray)

    X_train_seq = jnp.array(normalize_sequence(patchify_images(X_train_edges)), dtype=jnp.float32)
    X_test_seq = jnp.array(normalize_sequence(patchify_images(X_test_edges)), dtype=jnp.float32)
    y_train_bin = jnp.array(y_train_filtered.reshape(-1, 1), dtype=jnp.float32)
    y_test_bin = jnp.array(y_test_filtered.reshape(-1, 1), dtype=jnp.float32)

    return X_train_seq, y_train_bin, X_test_seq, y_test_bin

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist(n_train=12500, n_test=2500, patch_size=10, reduce_fn=np.max, classify_choice=[1, 7], show_size=True)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)