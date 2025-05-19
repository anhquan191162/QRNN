from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import jax.numpy as jnp
import numpy as np
def load_digits_data(n_train, encoding_type='angle'):
    """Load and preprocess digits"""
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y == 0) | (y == 1)
    X, y = X[np.where((y == 0) | (y == 1))], y[np.where((y == 0) | (y == 1))]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:100], y_test[:100]
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