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
import pandas as pd
import torch.nn.functional as F # Use updated MNIST_test.py
from data_loader import load_mnist, load_cifar2, load_digits_data
config.update("jax_enable_x64", True)

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
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_q = self.n_qub_enc * self.seq_num + self.anc_q
        dev = qml.device("default.qubit", wires=num_q)
        
        @qml.qnode(dev, interface="jax")
        def circuit(inputs, weights):
            index = 0
            for i in range(self.seq_num):
                start = i * self.n_qub_enc
                end = (i + 1) * self.n_qub_enc
                if self.encoding_type == 'angle':
                    for j in range(self.n_qub_enc):
                        qml.RY(inputs[start + j], j + self.anc_q)
                else:
                    raise ValueError('Unknown encoding type')
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

    def update_params(self, new_params):
        self.params = new_params

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true, y_pred):
    return jnp.mean((y_pred > 0.5) == y_true)

def make_forward_pass(circuit):
    batched_circuit = jax.vmap(circuit, in_axes=(0, None))
    @jax.jit
    def forward_pass(params, x):
        logits = batched_circuit(x, params)
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
        optax.clip(1.0),
        optax.adam(learning_rate=schedule)
    )

def create_batches(x, y, batch_size, key):
    n_samples = x.shape[0]
    indices = jax.random.permutation(key, n_samples)
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield x[batch_indices], y[batch_indices]

def train_qrnn(n_train, n_test, n_epochs, encoding_type='angle', dataset='digits', show=True, classify_choice=[0,1], batch_size=64):
    """Main training loop with mini-batch training."""
    if dataset == 'digits':
        from data_loader import load_digits_data
        x_train, x_test, y_train, y_test = load_digits_data(n_train, n_test, encoding_type)
    elif dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist(n_train, n_test, classify_choice=classify_choice)
    elif dataset == 'cifar2':
        from data_loader import load_cifar2
        x_train, y_train, x_test, y_test = load_cifar2(n_train, n_test)
    
    if dataset == 'digits':
        model = QRNN(anc_q=4, n_qub_enc=1, seq_num=4, D=2, encoding_type=encoding_type)
    elif dataset == 'mnist':
        model = QRNN(anc_q=3, n_qub_enc=1, seq_num=7, D=2, encoding_type=encoding_type)
    elif dataset == 'cifar2':
        model = QRNN(anc_q=3, n_qub_enc=1, seq_num=9, D=2, encoding_type=encoding_type)

    optimizer = create_optimizer()
    opt_state = optimizer.init(model.params)
    train_step = make_train_step(model.circuit, optimizer)
    evaluate = make_evaluate(model.circuit)

    train_losses = jnp.zeros(n_epochs)
    test_losses = jnp.zeros(n_epochs)
    train_accs = jnp.zeros(n_epochs)
    test_accs = jnp.zeros(n_epochs)

    total_time = 0
    print('Total parameters: {}'.format(model.params.shape[0]))
    print('Total trainable parameters: {}'.format(sum(p.size for p in model.params)))
    print('Total qubits: {}'.format(model.n_qub_enc * model.seq_num + model.anc_q))
    print('Total layers: {}'.format(model.D))
    print(f"Training for {n_epochs} epochs with batch size {batch_size}...")
    final_params = model.params

    for epoch in range(n_epochs):
        start_epoch = time.time()
        key = jax.random.PRNGKey(epoch)
        train_loss_epoch = 0.0
        num_batches = 0
        
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size, key):
            final_params, opt_state, batch_loss = train_step(final_params, opt_state, x_batch, y_batch)
            jax.lax.cond(num_batches % batch_size == 0 and show == True, lambda: debug.print("Batch {} Loss: {:.4f}", num_batches, batch_loss), lambda: None)
            train_loss_epoch += batch_loss
            num_batches += 1
        
        train_loss_epoch /= num_batches
        train_loss_val, train_acc_val = evaluate(final_params, x_train, y_train)
        test_loss_val, test_acc_val = evaluate(final_params, x_test, y_test)
        
        train_losses = train_losses.at[epoch].set(train_loss_epoch)
        test_losses = test_losses.at[epoch].set(test_loss_val)
        train_accs = train_accs.at[epoch].set(train_acc_val)
        test_accs = test_accs.at[epoch].set(test_acc_val)
        end_epoch = time.time()
        if show == True:
            debug.print("Epoch {}/{}", epoch + 1, n_epochs)
            debug.print("Train Loss: {:.4f}, Train Acc: {:.4f}", 
                        train_loss_val, train_acc_val)
            debug.print("Test Loss: {:.4f}, Test Acc: {:.4f}", 
                        test_loss_val, test_acc_val)
            debug.print("Epoch time: {:.2f} seconds", end_epoch - start_epoch)
        total_time += end_epoch - start_epoch
    model.update_params(final_params)
    debug.print("Training time: {:.2f} seconds", total_time)
    
    return train_losses, test_losses, train_accs, test_accs


if __name__ == "__main__":
    n_train = [1000]
    dataset = ['mnist']
    n_epochs = 100
    n_test = 100
    batch_size = 64
    gen_errs = []
    for n in n_train:
        train_losses, test_losses, train_accs, test_accs = train_qrnn(
            n, n_test, n_epochs, encoding_type='angle', dataset='mnist', show=True, classify_choice=[1,7], batch_size=batch_size
        )
        gen_err = (test_losses[-1] - train_losses[-1]) or np.mean(test_losses) - np.mean(train_losses) if test_losses[-1] - train_losses[-1] <= 0 else 0
        print(f"Training with {n} data points complete.") 
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}")
        print(f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}")
        print(f"Generalization Error: {gen_err:.4f}")
        
