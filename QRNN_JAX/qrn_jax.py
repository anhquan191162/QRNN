import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
from jax import config, debug
import torch
import torch.nn.functional as F
from data_loader import load_digits_data, load_mnist, load_cifar2
config.update("jax_enable_x64", True)
#AMPLITUDE ENCODING DOES NOT WORK 
# def amplitude_encode(data: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
#     """Amplitude‑encode a 1D array into a 2^n_qubits quantum state."""

#     # Compute target dimension
#     target_len = 2 ** n_qubits
#     data_len = data.shape[0]

#     # Truncate or pad
#     pad_len = jnp.maximum(0, target_len - data_len)
#     padded = jnp.concatenate([data, jnp.zeros(pad_len)], axis=0)
#     truncated = padded[:target_len]

#     # Normalize safely
#     norm = jnp.linalg.norm(truncated)
#     normalized = jnp.where(norm > 0, truncated / norm, jnp.ones_like(truncated) / jnp.sqrt(target_len))

#     return normalized

class QRNN:
    def __init__(self, anc_q, n_qub_enc, seq_num, D, encoding_type='angle'):
        self.anc_q = anc_q
        self.n_qub_enc = n_qub_enc
        self.seq_num = seq_num
        self.D = D
        self.encoding_type = encoding_type
        self.circuit = self._create_circuit()
        self.params = self._init_params()

    def _create_circuit(self):
        num_ansatz_q = self.anc_q + self.n_qub_enc #Number of qubits in the ansatz
        num_q = self.n_qub_enc * self.seq_num + self.anc_q #Number of qubits in the circuit
        dev = qml.device("default.qubit", wires=num_q)
        
        @qml.qnode(dev, interface="jax")
        def circuit(inputs, weights):
            index = 0
            for i in range(self.seq_num):
                start = i * self.n_qub_enc
                end = (i + 1) * self.n_qub_enc

                if self.encoding_type == 'angle':
                    # Angle encoding
                    for j in range(self.n_qub_enc):
                        qml.RY(inputs[start + j], j + self.anc_q)
                # elif self.encoding_type == 'amplitude':
                #     # Amplitude encoding
                #     qml.AmplitudeEmbedding(
                #                                 inputs[start:end],
                #                                 wires=range(self.anc_q, self.anc_q + self.n_qub_enc),
                #                                 pad_with=0.0,       # Pad with zeros
                #                                 normalize=True      # Normalize internally
                #                             )
                else:
                    raise ValueError('Unknown encoding type')

                # Ansatz
                num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
                block_weights = weights[i * num_para_per_bloc:(i + 1) * num_para_per_bloc]

                indx = 0
                for j in range(num_ansatz_q):
                    qml.RX(block_weights[indx], wires=j)
                    qml.RZ(block_weights[indx + 1], wires=j)
                    qml.RX(block_weights[indx + 2], wires=j)
                    indx += 3

                for d in range(self.D):
                    for j in range(num_ansatz_q):
                        qml.IsingZZ(block_weights[indx], wires=[j, (j + 1) % num_ansatz_q])
                        indx += 1
                    for j in range(num_ansatz_q):
                        qml.RY(block_weights[indx], wires=j)
                        indx += 1

                # SWAP state to next step
                if i != self.seq_num - 1:
                    for j in range(self.n_qub_enc):
                        q1 = j + self.anc_q
                        q2 = (i + 1) * self.n_qub_enc + j + self.anc_q
                        qml.SWAP(wires=[q1, q2])
            
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(num_ansatz_q)]))
        
        return circuit

    def _init_params(self):
        key = jax.random.PRNGKey(0)
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
        total_params = num_para_per_bloc * self.seq_num
        return jax.random.uniform(key, (total_params,), minval=-jnp.pi, maxval=jnp.pi)

    # def forward(self, x):
    #     # Handle both single and batch inputs
    #     if x.ndim == 1:
    #         return jax.nn.sigmoid(self.circuit(x, self.params))
    #     else:
    #         # Use vmap to vectorize the circuit over the batch dimension
    #         # batched_circuit = jax.vmap(self.circuit, in_axes=(0, None))
    #         self.batched_circuit = jax.jit(jax.vmap(self.circuit, in_axes=(0, None)))
    #         return jax.nn.sigmoid(self.batched_circuit(x, self.params))

    def update_params(self, new_params):
        self.params = new_params

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true, y_pred):
    """Accuracy"""
    return jnp.mean((y_pred > 0.5) == y_true)

# def load_digits_data(n_train, encoding_type='angle'):
#     """Load and preprocess digits"""
#     digits = load_digits()
#     X, y = digits.data, digits.target
#     mask = (y == 0) | (y == 1)
#     X, y = X[np.where((y == 0) | (y == 1))], y[np.where((y == 0) | (y == 1))]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)
#     X_train, y_train = X_train[:n_train], y_train[:n_train]
#     X_test, y_test = X_test[:100], y_test[:100]
#     X_train, X_test = X_train.reshape(-1,8,8), X_test.reshape(-1,8,8)

#     X_train = torch.tensor(X_train).unsqueeze(1).float()
#     X_test = torch.tensor(X_test).unsqueeze(1).float()

#     # Use smaller pooling size to preserve more information
#     pooled_train = F.adaptive_avg_pool2d(X_train, output_size=(3,3))
#     pooled_test = F.adaptive_avg_pool2d(X_test, output_size=(3,3))

#     train_seq = pooled_train.squeeze(1)
#     test_seq = pooled_test.squeeze(1)

#     # Scale to a smaller range for better quantum circuit performance
#     _min, _max = train_seq.min(), train_seq.max()
#     train_seq = ((train_seq - _min) / (_max - _min)) * torch.pi
#     test_seq = ((test_seq - _min) / (_max - _min)) * torch.pi

#     train = train_seq.reshape(-1, train_seq.shape[1] * train_seq.shape[2])
#     test = test_seq.reshape(-1, test_seq.shape[1] * test_seq.shape[2])
    
#     if encoding_type == 'angle':
#         return (
#             jnp.array(train.numpy(), dtype=jnp.float32),
#             jnp.array(test.numpy(), dtype=jnp.float32),
#             jnp.array(y_train, dtype=jnp.float32).reshape(-1,1),
#             jnp.array(y_test, dtype=jnp.float32).reshape(-1,1)
#         )
#     # elif encoding_type == 'amplitude':
#     #     # Amplitude encode the data
#     #     n_qubits = 4  # Using 4 qubits for encoding
#     #     train_encoded = jnp.array([amplitude_encode(x, n_qubits) for x in train.numpy()])
#     #     test_encoded = jnp.array([amplitude_encode(x, n_qubits) for x in test.numpy()])
#     #     return (
#     #         train_encoded,
#     #         test_encoded,
#     #         jnp.array(y_train, dtype=jnp.float32).reshape(-1,1),
#     #         jnp.array(y_test, dtype=jnp.float32).reshape(-1,1)
#     #     )
#     else:
#         raise ValueError('Unknown encoding type')

def make_forward_pass(circuit):
    # vectorize over the batch dimension
    batched_circuit = jax.vmap(circuit, in_axes=(0, None))
    # batched_circuit = jax.jit(jax.vmap(circuit, in_axes=(0, None)))

    @jax.jit
    def forward_pass(params, x):
        # shape(x) = (batch, features)
        logits = batched_circuit(x, params)       # -> (batch,)
        return jax.nn.sigmoid(logits).reshape(-1, 1)
    return forward_pass

def make_train_step(circuit, optimizer):
    forward_pass = make_forward_pass(circuit)
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            y_pred = forward_pass(p, x)
            return binary_cross_entropy(y, y_pred)
        
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        # print("Grad norm:", jnp.linalg.norm(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    
    return train_step

def make_evaluate(circuit):
    forward_pass = make_forward_pass(circuit)

    def evaluate(params, x, y):
        y_pred = forward_pass(params, x)
        loss = binary_cross_entropy(y, y_pred)
        acc = accuracy(y, y_pred)
        return loss, acc
    
    return evaluate

def create_optimizer(learning_rate=0.01):
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=1000,
        alpha=0.1
    )
    return optax.chain(
        optax.clip(1.0),  # Gradient clipping
        optax.adam(learning_rate=schedule)
    )

def train_qrnn(n_train, n_test, n_epochs,encoding_type='angle',dataset = 'digits', show = True):
    """Main training loop"""
    if dataset == 'digits':
        x_train, x_test, y_train, y_test = load_digits_data(n_train, n_test, encoding_type)
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist(n_train, n_test)
    elif dataset == 'cifar2':
        x_train, y_train, x_test, y_test = load_cifar2(n_train, n_test)
    # debug.print('Data loaded.')
    if dataset == 'digits':
        model = QRNN(anc_q=4, n_qub_enc=1, seq_num=4, D=2, encoding_type=encoding_type)
    elif dataset == 'mnist':
        model = QRNN(anc_q=3, n_qub_enc=1, seq_num=7, D=2, encoding_type=encoding_type) #3172, continue testing later
    elif dataset == 'cifar2':
        model = QRNN(anc_q=3, n_qub_enc=1, seq_num=9, D=2, encoding_type=encoding_type)
    # debug.print('Circuit and parameters initialized.')

    optimizer = create_optimizer()
    opt_state = optimizer.init(model.params)
    # debug.print('Optimizer initialized.')

    train_step = make_train_step(model.circuit, optimizer)
    evaluate = make_evaluate(model.circuit)

    # Initialize metrics arrays
    train_losses = jnp.zeros(n_epochs)
    test_losses = jnp.zeros(n_epochs)
    train_accs = jnp.zeros(n_epochs)
    test_accs = jnp.zeros(n_epochs)
    # debug.print('Train losses and test losses initialized.')

    start = time.time()
    # debug.print(f"Training for {n_epochs} epochs...")
    final_params = model.params

    for epoch in range(n_epochs):
        # Training step on full dataset
        final_params, opt_state, train_loss = train_step(final_params, opt_state, x_train, y_train)
        # debug.print(f"Train loss: {train_loss}")
        # Evaluate on full train and test sets
        train_loss_val, train_acc_val = evaluate(final_params, x_train, y_train)
        test_loss_val, test_acc_val = evaluate(final_params, x_test, y_test)
        # Store metrics
        train_losses = train_losses.at[epoch].set(train_loss_val)
        test_losses = test_losses.at[epoch].set(test_loss_val)
        train_accs = train_accs.at[epoch].set(train_acc_val)
        test_accs = test_accs.at[epoch].set(test_acc_val)

        # Print progress every 10 epochs
        if show:
            if (epoch + 1) % 50 == 0:
                debug.print("Epoch {}/{}", epoch + 1, n_epochs)
                debug.print("Train Loss: {:.4f}, Train Acc: {:.4f}", 
                            train_loss_val, train_acc_val)
                debug.print("Test Loss: {:.4f}, Test Acc: {:.4f}", 
                            test_loss_val, test_acc_val)
    
    model.update_params(final_params)
    debug.print("Training time: {:.2f} seconds", time.time() - start)
    
    return train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    n_train = [2, 5, 10, 20, 40, 80]
    dataset = ['digits', 'mnist', 'cifar2']
    n_epochs = 100
    n_test = 100
    for n in n_train:
        train_losses, test_losses, train_accs, test_accs = train_qrnn(
            n, n_test, n_epochs, encoding_type='angle', dataset = 'mnist', show = False
        )
        print(f"Training with {n} data points complete.") 
        print(f"Train Loss: {(train_losses[-1])}, Train Acc: {(train_accs[-1])}")
        print(f"Test Loss: {(test_losses[-1])}, Test Acc: {(test_accs[-1])}")
        print(f"Generalization Error: {(test_losses[-1]) - (train_losses[-1])}")
    # print("Train losses:", train_losses)
    # print("Test losses:", test_losses)
    # print("Train accuracies:", train_accs)
    # print("Test accuracies:", test_accs)
