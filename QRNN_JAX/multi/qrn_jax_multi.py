import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
from jax import config, debug
from dataloader_multiclass import load_mnist
config.update("jax_enable_x64", True)

class QRNN:
    def __init__(self, anc_q, n_qub_enc, seq_num, D, num_classes=10, encoding_type='amplitude'):
        self.anc_q = anc_q
        self.n_qub_enc = n_qub_enc
        self.seq_num = seq_num
        self.D = D
        self.num_classes = num_classes
        self.encoding_type = encoding_type
        self.circuit = self._create_circuit()
        self.params = self._init_params()

    def _create_circuit(self):
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_q = self.n_qub_enc * self.seq_num + self.anc_q
        dev = qml.device("default.qubit", wires=num_q)
        
        @qml.qnode(dev, interface="jax", diff_method=qml.gradients.finite_diff, gradient_kwargs={'h': 1e-3})
        def circuit(inputs, weights):
            for i in range(self.seq_num):
                start = i * self.n_qub_enc
                end = (i + 1) * self.n_qub_enc

                if self.encoding_type == 'amplitude':
                    # Amplitude encoding for 4 features (requires 2 qubits)
                    feature_vec = inputs[:4]  # Take all 4 features (seq_num=1)
                    qml.StatePrep(feature_vec, wires=range(self.anc_q, self.anc_q + self.n_qub_enc))
                elif self.encoding_type == 'angle':
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

                # No SWAP gates needed for seq_num=1
                if i != self.seq_num - 1:
                    for j in range(self.n_qub_enc):
                        q1 = j + self.anc_q
                        q2 = (i + 1) * self.n_qub_enc + j + self.anc_q
                        qml.SWAP(wires=[q1, q2])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_classes)]
        
        return circuit

    def _init_params(self):
        key = jax.random.PRNGKey(0)
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
        total_params = num_para_per_bloc * self.seq_num
        return jax.random.uniform(key, (total_params,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)

    def update_params(self, new_params):
        self.params = new_params

def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    return jnp.mean(jnp.argmax(y_pred, axis=1) == jnp.argmax(y_true, axis=1))

def make_forward_pass(circuit, num_classes):
    batched_circuit = jax.vmap(circuit, in_axes=(0, None))

    @jax.jit
    def forward_pass(params, x):
        logits = batched_circuit(x, params)
        logits = jnp.stack(logits, axis=1)
        return jax.nn.softmax(logits)
    return forward_pass

def make_train_step(circuit, optimizer, num_classes):
    forward_pass = make_forward_pass(circuit, num_classes)
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            y_pred = forward_pass(p, x)
            return categorical_cross_entropy(y, y_pred)
        
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.linalg.norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, grad_norm
    
    return train_step

def make_evaluate(circuit, num_classes):
    forward_pass = make_forward_pass(circuit, num_classes)

    def evaluate(params, x, y):
        y_pred = forward_pass(params, x)
        loss = categorical_cross_entropy(y, y_pred)
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

def train_qrnn(n_train, n_test, n_epochs, encoding_type='amplitude', num_classes=10, batch_size=4, show=True):
    x_train, y_train, x_test, y_test = load_mnist(
        n_train=n_train, n_test=n_test, seq_len=4, num_classes=num_classes
    )
    
    model = QRNN(anc_q=2, n_qub_enc=2, seq_num=4, D=1, num_classes=num_classes, encoding_type=encoding_type)
    optimizer = create_optimizer()
    opt_state = optimizer.init(model.params)
    train_step = make_train_step(model.circuit, optimizer, num_classes)
    evaluate = make_evaluate(circuit=model.circuit, num_classes=num_classes)

    train_losses = jnp.zeros(n_epochs, dtype=jnp.float64)
    test_losses = jnp.zeros(n_epochs, dtype=jnp.float64)
    train_accs = jnp.zeros(n_epochs, dtype=jnp.float64)
    test_accs = jnp.zeros(n_epochs, dtype=jnp.float64)

    start = time.time()
    final_params = model.params

    num_batches = (n_train + batch_size - 1) // batch_size
    print(f"Number of batches: {num_batches}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of training data: {n_train}")
    print(f"Number of test data: {n_test}")
    print(f"Number of classes: {num_classes}")
    print(f"Encoding type: {encoding_type}")
    print(f"Number of parameters: {model.params.shape[0]}")
    print(f"Number of qubits: {model.n_qub_enc}")
    print(f"Number of ancilla qubits: {model.anc_q}")
    print(f"Number of blocks: {model.seq_num}")
    print(f"Number of parameters per block: {model.D}")
    print(f"Number of total qubits: {model.n_qub_enc * model.seq_num + model.anc_q}")
    print('--------------------------------')
    print('Training...')
    for epoch in range(n_epochs):
        epoch_start = time.time()
        key = jax.random.PRNGKey(epoch)
        indices = jax.random.permutation(key, n_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_train)
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            batch_start = time.time()
            final_params, opt_state, batch_loss, grad_norm = train_step(final_params, opt_state, x_batch, y_batch)
            batch_time = time.time() - batch_start
            epoch_loss += batch_loss * (end_idx - start_idx) / n_train
            epoch_grad_norm += grad_norm * (end_idx - start_idx) / n_train

            if show and (epoch + 1) % 1 == 0:
                debug.print("Batch {}/{}, time: {:.2f} seconds, loss: {:.4f}, grad_norm: {:.4f}",
                            i + 1, num_batches, batch_time, batch_loss, grad_norm)

        train_loss_val, train_acc_val = evaluate(final_params, x_train, y_train)
        test_loss_val, test_acc_val = evaluate(final_params, x_test, y_test)

        train_losses = train_losses.at[epoch].set(train_loss_val)
        test_losses = test_losses.at[epoch].set(test_loss_val)
        train_accs = train_accs.at[epoch].set(train_acc_val)
        test_accs = test_accs.at[epoch].set(test_acc_val)

        if show and (epoch + 1) % 1 == 0:
            debug.print("Epoch {}/{}", epoch + 1, n_epochs)
            debug.print("Train Loss: {:.4f}, Train Acc: {:.4f}", 
                        train_loss_val, train_acc_val)
            debug.print("Test Loss: {:.4f}, Test Acc: {:.4f}", 
                        test_loss_val, test_acc_val)
            debug.print("Epoch time: {:.2f} seconds, Avg Grad Norm: {:.4f}", 
                        time.time() - epoch_start, epoch_grad_norm)
    
    model.update_params(final_params)
    debug.print("Total training time: {:.2f} seconds", time.time() - start)
    
    return train_losses, test_losses, train_accs, test_accs

if __name__ == "__main__":
    n_train = [10000]
    n_epochs = 100
    n_test = 100
    num_classes = 10
    batch_size = 128
    for n in n_train:
        train_losses, test_losses, train_accs, test_accs = train_qrnn(
            n, n_test, n_epochs, encoding_type='amplitude', num_classes=num_classes, batch_size=batch_size, show=True
        )
        print(f"Training with {n} data points complete.") 
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}")
        print(f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}")
        print(f"Generalization Error: {(test_losses[-1] - train_losses[-1]):.4f}")