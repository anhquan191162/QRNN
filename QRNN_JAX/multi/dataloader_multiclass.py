import numpy as np
import jax.numpy as jnp
from jax.nn import one_hot
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import os
from sklearn.linear_model import LogisticRegression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)

def build_cnn_classifier(output_dim=4):
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    features = layers.Dense(output_dim, activation='tanh', name='feature_layer')(x)
    outputs = layers.Dense(10, activation='softmax')(features)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_and_save_cnn(x_train, y_train, output_dim=4, epochs=5, save_path='cnn_model.h5'):
    print(f"Training CNN for {epochs} epochs...")
    model = build_cnn_classifier(output_dim)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1)
    model.save(save_path)
    print(f"CNN saved to {save_path}")

def load_cnn_feature_model(save_path='cnn_model.h5', output_dim=4):
    full_model = tf.keras.models.load_model(save_path)
    feature_model = models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer('feature_layer').output
    )
    feature_model.trainable = False
    return feature_model

def test_cnn_features(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.array(X_train), np.array(y_train))
    return clf.score(np.array(X_train), np.array(y_train))

def load_mnist(n_train=15000, n_test=10000, seq_len=4, num_classes=10,
               train_cnn_epochs=None, force_retrain=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    idx_train = np.random.choice(len(x_train), n_train, replace=False)
    idx_test = np.random.choice(len(x_test), n_test, replace=False)
    x_train, y_train = x_train[idx_train], y_train[idx_train]
    x_test, y_test = x_test[idx_test], y_test[idx_test]

    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if train_cnn_epochs is None:
        if n_train <= 1000 and n_test <= 200:
            train_cnn_epochs = 100
        elif n_train <= 5000 and n_test <= 1000:
            train_cnn_epochs = 80
        elif n_train <= 10000 and n_test <= 2000:
            train_cnn_epochs = 40
        else:
            train_cnn_epochs = 20

    cnn_model_path = f'cnn_model_{n_train}train_{train_cnn_epochs}ep.h5'

    if force_retrain or not os.path.exists(cnn_model_path):
        train_and_save_cnn(x_train, y_train, output_dim=seq_len,
                           epochs=train_cnn_epochs, save_path=cnn_model_path)

    feature_model = load_cnn_feature_model(cnn_model_path, output_dim=seq_len)
    X_train_feat = feature_model.predict(x_train, verbose=0)
    X_test_feat = feature_model.predict(x_test, verbose=0)

    print(f"Logistic Regression accuracy on CNN features: {test_cnn_features(X_train_feat, y_train):.3f}")

    # Normalize features for amplitude encoding
    X_train_norm = np.linalg.norm(X_train_feat, axis=1, keepdims=True)
    X_test_norm = np.linalg.norm(X_test_feat, axis=1, keepdims=True)
    X_train_feat = X_train_feat / np.where(X_train_norm > 0, X_train_norm, 1.0)  # Avoid division by zero
    X_test_feat = X_test_feat / np.where(X_test_norm > 0, X_test_norm, 1.0)

    X_train_jax = jnp.array(X_train_feat, dtype=jnp.float64)
    X_test_jax = jnp.array(X_test_feat, dtype=jnp.float64)
    y_train_jax = one_hot(jnp.array(y_train, dtype=jnp.int32), num_classes)
    y_test_jax = one_hot(jnp.array(y_test, dtype=jnp.int32), num_classes)

    return X_train_jax, y_train_jax, X_test_jax, y_test_jax

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist(
        n_train=1000, n_test=200, seq_len=4,
        train_cnn_epochs=None, force_retrain=True
    )
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)