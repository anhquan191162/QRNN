from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import jax.numpy as jnp
import numpy as np
import keras 
from scipy.ndimage import sobel
from data_loader import load_cifar2
# def load_cifar2(n_train = 100, n_test = 100, patch_size = 11, reduce_fn = np.mean):
#     def rgb2gray(images):
#     # Uses ITU-R BT.601 luma transform
#         return 0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2]
#     def sobel_edges(images):
#     # images: (N, H, W)
#         edge_images = []
#         for img in images:
#             dx = sobel(img, axis=0, mode='constant')
#             dy = sobel(img, axis=1, mode='constant')
#             mag = np.hypot(dx, dy)  # magnitude of gradient
#             edge_images.append(mag)
#         return np.array(edge_images)
#     def patchify_images(images):
#         H, W = images.shape[1:]
#         sequences = []
#         for img in images:
#             seq = [
#                 reduce_fn(img[i:i+patch_size, j:j+patch_size])
#                 for i in range(0, H, patch_size)
#                 for j in range(0, W, patch_size)
#             ]
#             sequences.append(seq)
#         return np.array(sequences)

#     def normalize_sequence(seq, min_val=0.0, max_val=255.0):
#         return (seq - min_val) / (max_val - min_val) * np.pi
#     (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#     y_train = y_train.squeeze() # (N, 1) -> (N,)
#     y_test = y_test.squeeze() # (N, 1) -> (N,)

#     train_mask = (y_train == 0) | (y_train == 1)
#     test_mask = (y_test == 0) | (y_test == 1)
#     np.random.seed(42)
#     train_indices = np.random.choice(len(x_train[train_mask]), size=n_train, replace=False)
#     test_indices = np.random.choice(len(x_test[test_mask]), size=n_test, replace=False)

#     x_train_filtered = x_train[train_mask][train_indices]
#     y_train_filtered = y_train[train_mask][train_indices]
#     x_test_filtered = x_test[test_mask][test_indices]
#     y_test_filtered = y_test[test_mask][test_indices]
    

#     y_train_filtered = y_train_filtered.reshape(-1, 1)
#     y_test_filtered = y_test_filtered.reshape(-1, 1)

#     X_train_gray = rgb2gray(x_train_filtered)
#     X_test_gray = rgb2gray(x_test_filtered)

#     X_train_edges = sobel_edges(X_train_gray)
#     X_test_edges = sobel_edges(X_test_gray)

#     X_train_seq = jnp.array(normalize_sequence(patchify_images(X_train_edges)), dtype=jnp.float32)
#     X_test_seq = jnp.array(normalize_sequence(patchify_images(X_test_edges)), dtype=jnp.float32)
#     y_train_bin = jnp.array(y_train_filtered, dtype=jnp.float32)
#     y_test_bin = jnp.array(y_test_filtered, dtype=jnp.float32)

#     return X_train_seq, y_train_bin, X_test_seq, y_test_bin

if __name__ == "__main__":
    X_train_seq, y_train_bin, X_test_seq, y_test_bin = load_cifar2()
    print(X_train_seq.shape)
    print(X_test_seq.shape)
    print(y_train_bin.shape)
    print(y_test_bin.shape)
    print(X_train_seq[0])
    print(X_test_seq[0])
    print(y_train_bin[0])
    print(y_test_bin[0])