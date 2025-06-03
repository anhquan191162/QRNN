import os
import shutil
from typing import List, Tuple, Callable, Dict
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def binary_cross_entropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    eps = 1e-15
    y_pred = jnp.clip(y_pred, eps, 1 - eps)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    return jnp.mean((y_pred > 0.5) == y_true)

def make_forward_pass(circuit: Callable) -> Callable:
    batched_circuit = jax.vmap(circuit, in_axes=(0, None))
    @jax.jit
    def forward_pass(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        logits = batched_circuit(x, params)
        return jax.nn.sigmoid(logits).reshape(-1, 1)
    return forward_pass

def make_train_step(circuit: Callable, optimizer: optax.GradientTransformation) -> Callable:
    forward_pass = make_forward_pass(circuit)
    
    @jax.jit
    def train_step(params: jnp.ndarray, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, optax.OptState, float]:
        def loss_fn(p: jnp.ndarray) -> float:
            y_pred = forward_pass(p, x)
            return binary_cross_entropy(y, y_pred)
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    
    return train_step

def make_evaluate(circuit: Callable) -> Callable:
    forward_pass = make_forward_pass(circuit)
    def evaluate(params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[float, float]:
        y_pred = forward_pass(params, x)
        return binary_cross_entropy(y, y_pred), accuracy(y, y_pred)
    return evaluate

def create_optimizer(learning_rate: float = 0.01) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=1000, alpha=0.1)
    return optax.chain(optax.clip(1.0), optax.adam(learning_rate=schedule))

def create_batches(x: jnp.ndarray, y: jnp.ndarray, batch_size: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_samples = x.shape[0]
    indices = jax.random.permutation(key, n_samples)
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield x[batch_indices], y[batch_indices]

def FGSM(model_fn: Callable, params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, eps: float) -> jnp.ndarray:
    loss_fn = lambda x_in: binary_cross_entropy(y, model_fn(params, x_in))
    grad = jax.grad(loss_fn)(x)
    return jnp.clip(x + eps * jnp.sign(grad), 0.0, 1.0)

def PGD(model_fn: Callable, params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, eps: float = 8/255, alpha: float = 2/255, steps: int = 5) -> jnp.ndarray:
    x_adv = x.copy()
    loss_fn = lambda x_in: binary_cross_entropy(y, model_fn(params, x_in))
    for _ in range(steps):
        grads = jax.grad(loss_fn)(x_adv)
        x_adv = x_adv + alpha * jnp.sign(grads)
        x_adv = jnp.clip(x_adv, x - eps, x + eps)
        x_adv = jnp.clip(x_adv, 0.0, 1.0)
    return x_adv

def APGD(model_fn: Callable, params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, eps: float = 0.3, alpha: float = 0.01, steps: int = 10, seed: int = 0) -> jnp.ndarray:
    loss_fn = lambda x_in: binary_cross_entropy(y, model_fn(params, x_in))
    x_adv = x.copy()
    step_size = alpha
    cur_loss = loss_fn(x_adv)
    best_loss = cur_loss
    best_adv = x_adv

    for _ in range(steps):
        grads = jax.grad(loss_fn)(x_adv)
        x_adv = x_adv + step_size * jnp.sign(grads)
        x_adv = jnp.clip(x_adv, x - eps, x + eps)
        x_adv = jnp.clip(x_adv, 0.0, 1.0)
        cur_loss = loss_fn(x_adv)
        if cur_loss < best_loss:
            step_size *= 0.75
        update_mask = (cur_loss > best_loss).reshape(-1, 1)
        best_adv = jnp.where(update_mask, x_adv, best_adv)
        best_loss = jnp.maximum(best_loss, cur_loss)
    return best_adv

def MIM(model_fn: Callable, params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 1.0) -> jnp.ndarray:
    x_adv = x.copy()
    momentum = jnp.zeros_like(x)
    loss_fn = lambda x_in: binary_cross_entropy(y, model_fn(params, x_in))
    for _ in range(steps):
        grad = jax.grad(loss_fn)(x_adv)
        grad = grad / (jnp.mean(jnp.abs(grad)) + 1e-8)
        momentum = decay * momentum + grad
        x_adv = x_adv + alpha * jnp.sign(momentum)
        x_adv = jnp.clip(x_adv, x - eps, x + eps)
        x_adv = jnp.clip(x_adv, 0.0, 1.0)
    return x_adv

def SA(model_fn: Callable, params: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, eps: float = 0.3, steps: int = 10, p: float = 0.05, seed: int = 0) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    adv_x = jnp.array(x, dtype=jnp.float32)
    origin_x = jnp.array(x, dtype=jnp.float32)
    input_dim = x.shape[1]
    loss_fn = lambda x_in: binary_cross_entropy(y, model_fn(params, x_in))
    best_loss = loss_fn(adv_x)
    successful_mask = jnp.zeros(x.shape[0], dtype=jnp.bool_)

    @jax.jit
    def perturb_sample(idx: int, img: jnp.ndarray, origin_img: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, bool, jax.random.PRNGKey]:
        current_logits = model_fn(params, img[None])
        current_pred = (current_logits > 0.5).astype(jnp.int32).squeeze()
        true_label = y[idx].astype(jnp.int32).squeeze()

        def successful_body(_): return img, True, key
        def perturb_body(key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, bool, jax.random.PRNGKey]:
            num_elements = int(p * input_dim)
            segment_len = max(1, num_elements)
            key, key_start, key_pert = jax.random.split(key, 3)
            start_idx = jax.random.randint(key_start, (), 0, input_dim - segment_len + 1)
            origin_patch = jax.lax.dynamic_slice(origin_img, (start_idx,), (segment_len,))
            perturbation = jax.random.uniform(key_pert, (segment_len,), minval=-eps, maxval=eps, dtype=jnp.float32)
            pert_patch = origin_patch + perturbation
            cand_img = jax.lax.dynamic_update_slice(img, pert_patch, (start_idx,))
            cand_img = jnp.clip(cand_img, origin_img - eps, origin_img + eps)
            cand_img = jnp.clip(cand_img, 0.0, 1.0)
            return cand_img, False, key

        is_successful = current_pred != true_label
        return jax.lax.cond(is_successful, successful_body, perturb_body, key)

    for i in range(steps):
        if jnp.all(successful_mask): break
        adv_x_cur_iter = adv_x.copy()

        def scan_body(carry: Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey], idx: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey], None]:
            adv_x_tmp, successful_mask_tmp, key = carry
            img = adv_x_tmp[idx]
            origin_img = origin_x[idx]
            cand_img, is_successful, new_key = perturb_sample(idx, img, origin_img, key)
            adv_x_tmp = adv_x_tmp.at[idx].set(cand_img)
            successful_mask_tmp = successful_mask_tmp.at[idx].set(is_successful)
            return (adv_x_tmp, successful_mask_tmp, new_key), None

        (adv_x_cur_iter, successful_mask, key), _ = jax.lax.scan(scan_body, (adv_x_cur_iter, successful_mask, key), jnp.arange(x.shape[0]))
        candidate_loss = loss_fn(adv_x_cur_iter)
        improved_mask = (candidate_loss > best_loss).reshape(-1, 1)
        adv_x = jnp.where(improved_mask, adv_x_cur_iter, adv_x)
        best_loss = jnp.where(improved_mask, candidate_loss, best_loss)

    return adv_x

def generalization_error(train_losses: float, test_losses: float) -> float:
    return test_losses - train_losses

def save_to_csv(df: pd.DataFrame, path: str, mode: str, header: bool) -> None:
    """Save DataFrame to CSV file."""
    df.to_csv(path, mode=mode, header=header, index=False)
    print(f"Results saved to {path}")

def plot_confusion_matrix(y_true: jnp.ndarray, y_pred: jnp.ndarray, dataset: str, classify_choice: List[int], main_folder: str, desc: str, n_train: int, log: bool) -> None:
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset} - Classes {classify_choice[0]} vs {classify_choice[1]}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if log:
        os.makedirs(f'{main_folder}/confusion_matrix', exist_ok=True)
        cm_path = f'{main_folder}/confusion_matrix/confusion_matrix_{dataset}_ntrain{n_train}_{classify_choice[0]}_{classify_choice[1]}_{desc}.png'
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")
    else:
        plt.show()

def plot_multi_barplot(data: List[Dict], metric: str, title: str, y_label: str, main_folder: str, filename: str, log: bool = True) -> None:
    """Plot a multi-barplot for the specified metric across datasets."""
    if not data or any(result is None for result in data):
        print(f"No results to plot for {metric}.")
        return

    plot_data = []
    attack_types = ['Clean'] + [m['Attack'] for m in data[0]['adv_metrics']] if metric == 'Accuracy' else [m['Attack'] for m in data[0]['adv_metrics']]
    
    for result in data:
        dataset = result['dataset_name']
        if metric == 'Accuracy':
            clean_acc = result['clean_acc']
            adv_accs = [1.0 - m['Attack_Success_Rate'] for m in result['adv_metrics']]
            values = [clean_acc] + adv_accs if clean_acc is not None else adv_accs
        else:
            values = [m['Attack_Success_Rate'] for m in result['adv_metrics']]
        
        for attack, value in zip(attack_types, values):
            plot_data.append({'Dataset': dataset, metric: float(value), 'Attack Type': attack})

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    sns.barplot(x='Dataset', y=metric, hue='Attack Type', data=df)
    plt.title(title)
    plt.xlabel('Dataset')
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Attack Type')
    
    if log:
        os.makedirs(f'{main_folder}/adversarial_results/plots', exist_ok=True)
        plt.savefig(f'{main_folder}/adversarial_results/plots/{filename}')
        plt.close()
        print(f"Multi-barplot for {metric} saved to {main_folder}/adversarial_results/plots/{filename}")
    else:
        plt.show()