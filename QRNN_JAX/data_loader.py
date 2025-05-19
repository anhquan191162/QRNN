from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import jax.numpy as jnp
import numpy as np
import keras 
from scipy.ndimage import sobel
from skimage.exposure import equalize_hist
def load_digits_data(n_train, n_test = 100, encoding_type='angle'):
    """Load and preprocess digits"""
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y == 0) | (y == 1)
    X, y = X[np.where((y == 0) | (y == 1))], y[np.where((y == 0) | (y == 1))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]
    X_train, X_test = X_train.reshape(-1,8,8), X_test.reshape(-1,8,8)

    X_train = torch.tensor(X_train).unsqueeze(1).float()
    X_test = torch.tensor(X_test).unsqueeze(1).float()

    # Use smaller pooling size to preserve more information
    pooled_train = F.adaptive_avg_pool2d(X_train, output_size=(3,3))
    pooled_test = F.adaptive_avg_pool2d(X_test, output_size=(3,3))

    train_seq = pooled_train.squeeze(1)
    test_seq = pooled_test.squeeze(1)

    # Scale to a smaller range for better quantum circuit performance
    _min, _max = train_seq.min(), train_seq.max()
    train_seq = ((train_seq - _min) / (_max - _min)) * torch.pi
    test_seq = ((test_seq - _min) / (_max - _min)) * torch.pi

    train = train_seq.reshape(-1, train_seq.shape[1] * train_seq.shape[2])
    test = test_seq.reshape(-1, test_seq.shape[1] * test_seq.shape[2])
    
    if encoding_type == 'angle':
        return (
            jnp.array(train.numpy(), dtype=jnp.float32),
            jnp.array(test.numpy(), dtype=jnp.float32),
            jnp.array(y_train, dtype=jnp.float32).reshape(-1,1),
            jnp.array(y_test, dtype=jnp.float32).reshape(-1,1)
        )
    # elif encoding_type == 'amplitude':
    #     # Amplitude encode the data
    #     n_qubits = 4  # Using 4 qubits for encoding
    #     train_encoded = jnp.array([amplitude_encode(x, n_qubits) for x in train.numpy()])
    #     test_encoded = jnp.array([amplitude_encode(x, n_qubits) for x in test.numpy()])
    #     return (
    #         train_encoded,
    #         test_encoded,
    #         jnp.array(y_train, dtype=jnp.float32).reshape(-1,1),
    #         jnp.array(y_test, dtype=jnp.float32).reshape(-1,1)
    #     )
    else:
        raise ValueError('Unknown encoding type')
    

def load_mnist(n_train=100,n_test=100, patch_size=10, reduce_fn=np.max):
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
        return np.array(sequences)

    def normalize_sequence(seq, min_val=0.0, max_val=255.0):
        return (seq - min_val) / (max_val - min_val) * np.pi

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, y_train = X_train[(y_train == 0) | (y_train == 1)], y_train[(y_train == 0) | (y_train == 1)]
    X_test, y_test = X_test[(y_test == 0) | (y_test == 1)], y_test[(y_test == 0) | (y_test == 1)]

    np.random.seed(42)
    train_indices = np.random.choice(len(X_train), size=n_train, replace=False)
    test_indices = np.random.choice(len(X_test), size=n_test, replace=False)

    X_train_sampled = X_train[train_indices]
    y_train_sampled = y_train[train_indices]
    X_test_sampled = X_test[test_indices]
    y_test_sampled = y_test[test_indices]

    X_train_seq = jnp.array(normalize_sequence(patchify_images(X_train_sampled)), dtype=jnp.float32)
    X_test_seq = jnp.array(normalize_sequence(patchify_images(X_test_sampled)), dtype=jnp.float32)
    y_train_bin = jnp.array(y_train_sampled.reshape(-1, 1), dtype=jnp.float32)
    y_test_bin = jnp.array(y_test_sampled.reshape(-1, 1), dtype=jnp.float32)

    return X_train_seq, y_train_bin, X_test_seq, y_test_bin


def load_cifar2(n_train = 100, n_test = 100, patch_size = 11, reduce_fn = np.mean):
    def rgb2gray(images):
    # Uses ITU-R BT.601 luma transform
        return 0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2]
    def sobel_edges(images):
    # images: (N, H, W)
        edge_images = []
        for img in images:
            dx = sobel(img, axis=0, mode='constant')
            dy = sobel(img, axis=1, mode='constant')
            mag = np.hypot(dx, dy)  # magnitude of gradient
            edge_images.append(mag)
        return np.array(edge_images)
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
        return np.array(sequences)

    
    def enhance_contrast(images):
        return np.array([equalize_hist(img) for img in images])
    
    def normalize_sequence(seq):
        min_val = seq.min()
        max_val = seq.max()
        return (seq - min_val) / (max_val - min_val + 1e-8) * np.pi
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.squeeze() # (N, 1) -> (N,)
    y_test = y_test.squeeze() # (N, 1) -> (N,)

    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test == 0) | (y_test == 1)
    np.random.seed(42)
    train_indices = np.random.choice(len(x_train[train_mask]), size=n_train, replace=False)
    test_indices = np.random.choice(len(x_test[test_mask]), size=n_test, replace=False)

    x_train_filtered = x_train[train_mask][train_indices]
    y_train_filtered = y_train[train_mask][train_indices]
    x_test_filtered = x_test[test_mask][test_indices]
    y_test_filtered = y_test[test_mask][test_indices]
    

    y_train_filtered = y_train_filtered.reshape(-1, 1)
    y_test_filtered = y_test_filtered.reshape(-1, 1)

    X_train_gray = rgb2gray(x_train_filtered)
    X_test_gray = rgb2gray(x_test_filtered)

    X_train_gray_enhanced = enhance_contrast(X_train_gray)
    X_test_gray_enhanced = enhance_contrast(X_test_gray)

    X_train_edges = sobel_edges(X_train_gray_enhanced)
    X_test_edges = sobel_edges(X_test_gray_enhanced)

    X_train_seq = jnp.array(normalize_sequence(patchify_images(X_train_edges)), dtype=jnp.float32)
    X_test_seq = jnp.array(normalize_sequence(patchify_images(X_test_edges)), dtype=jnp.float32)
    y_train_bin = jnp.array(y_train_filtered, dtype=jnp.float32)
    y_test_bin = jnp.array(y_test_filtered, dtype=jnp.float32)

    return X_train_seq, y_train_bin, X_test_seq, y_test_bin