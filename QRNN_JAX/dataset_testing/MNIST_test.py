import keras
import numpy as np
import jax.numpy as jnp

def load_mnist(n_train=100, n_test=100, patch_size=10, reduce_fn=np.max, classify_choice = [0,1], show_size = False):
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
    # Filter for digits 1 and 7
    train_mask = (y_train == classify_choice[0]) | (y_train == classify_choice[1])
    test_mask = (y_test == classify_choice[0]) | (y_test == classify_choice[1]) 
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    if show_size:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
    # Map labels: 1 -> 0, 7 -> 1
    y_train = np.where(y_train == classify_choice[0], 0, 1)
    y_test = np.where(y_test == classify_choice[0], 0, 1)

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


if __name__ == "__main__":
    X_train_seq, y_train_bin, X_test_seq, y_test_bin = load_mnist(8000, 2000, show_size=True)
    print(X_train_seq.shape)
    print(X_test_seq.shape)
    print(y_train_bin.shape)
    print(y_test_bin.shape)
    print('Example of X_train_seq:', X_train_seq[0])
    print('Example of y_train_bin:', y_train_bin[0])
    