import keras
import numpy as np
import jax.numpy as jnp
# np.random.seed(42)
def load_mnist(n_train=15000,n_test=150, patch_size=10, reduce_fn=np.max):
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




if __name__ == "__main__":
    X_train_seq, y_train_bin, X_test_seq, y_test_bin = load_mnist(80)
    print(X_train_seq.shape)
    print(X_test_seq.shape)
    print(y_train_bin.shape)
    print(y_test_bin.shape)
    print('Example of X_train_seq:', X_train_seq[0])
    print('Example of y_train_bin:', y_train_bin[0])
    print(X_train_seq[1])
    print(X_train_seq[2])


