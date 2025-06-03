import os
import shutil
from typing import List, Tuple, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from jax import config
from data_loader import load_mnist, load_cifar2, load_old_cifar2
from gen_plot import gen_plotter2
from qrnn_utils import (
    binary_cross_entropy, accuracy, make_forward_pass, make_train_step, make_evaluate,
    create_optimizer, create_batches, FGSM, PGD, APGD, MIM, SA,
    generalization_error, save_to_csv, plot_confusion_matrix, plot_multi_barplot
)

config.update("jax_enable_x64", False)

class QRNN:
    def __init__(self, anc_q: int, n_qub_enc: int, seq_num: int, D: int, encoding_type: str = 'angle'):
        self.anc_q = anc_q
        self.n_qub_enc = n_qub_enc
        self.seq_num = seq_num
        self.D = D
        self.encoding_type = encoding_type
        self.circuit = self._create_circuit()
        self.params = self._init_params()

    def _create_circuit(self) -> callable:
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_q = self.n_qub_enc * self.seq_num + self.anc_q
        dev = qml.device("default.qubit", wires=num_q)

        @qml.qnode(dev, interface="jax")
        def circuit(inputs: jnp.ndarray, weights: jnp.ndarray) -> float:
            for i in range(self.seq_num):
                start = i * self.n_qub_enc
                if self.encoding_type == 'angle':
                    for j in range(self.n_qub_enc):
                        qml.RY(inputs[start + j], j + self.anc_q)
                else:
                    raise ValueError(f"Unknown encoding type: {self.encoding_type}")
                
                num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
                block_weights = weights[i * num_para_per_bloc:(i + 1) * num_para_per_bloc]
                idx = 0
                for j in range(num_ansatz_q):
                    qml.RX(block_weights[idx], wires=j)
                    qml.RZ(block_weights[idx + 1], wires=j)
                    qml.RX(block_weights[idx + 2], wires=j)
                    idx += 3
                for d in range(self.D):
                    for j in range(num_ansatz_q):
                        qml.IsingZZ(block_weights[idx], wires=[j, (j + 1) % num_ansatz_q])
                        idx += 1
                    for j in range(num_ansatz_q):
                        qml.RY(block_weights[idx], wires=j)
                        idx += 1
                if i != self.seq_num - 1:
                    for j in range(self.n_qub_enc):
                        qml.SWAP(wires=[j + self.anc_q, (i + 1) * self.n_qub_enc + j + self.anc_q])
            return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(num_ansatz_q)]))
        
        return circuit

    def _init_params(self) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)
        num_ansatz_q = self.anc_q + self.n_qub_enc
        num_para_per_bloc = num_ansatz_q * (3 * self.D + 2)
        total_params = num_para_per_bloc * self.seq_num
        return jax.random.uniform(key, (total_params,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)

    def update_params(self, new_params: jnp.ndarray) -> None:
        self.params = new_params

def clear_dir() -> None:
    """Manually clear all contents of QRNN_JAX directories."""
    main_folder = 'QRNN_JAX'
    subdirs = ['generalization_results', 'top_error_results', 'adversarial_results', 'plots', 'confusion_matrix']
    for subdir in subdirs:
        path = os.path.join(main_folder, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Cleared directory: {path}")
        os.makedirs(path, exist_ok=True)
        print(f"Recreated directory: {path}")

def get_batch_size(n_train: int) -> int:
    """Determine batch size based on training size."""
    if n_train <= 1000:
        return 64
    elif n_train <= 5000:
        return 128
    elif n_train <= 10000:
        return 256
    else:
        return 512

def train_and_evaluate_adversarial(
    n_train: int = 100,
    n_test: int = 100,
    n_epochs: int = 10,
    dataset: str = 'mnist',
    classify_choice: List[int] = [1, 7],
    feature_size: str = '2x2',
    desc: str = '',
    evaluate_adv: bool = True,
    mode: str = 'a',
    log: bool = True,
    clear_dir_auto: bool = False ) -> Optional[Dict]:
    main_folder = 'QRNN_JAX'
    header = mode == 'w'
    
    if clear_dir_auto and log and header:
        clear_dir()

    # Load dataset
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist(n_train, n_test, classify_choice=classify_choice, show_size=False)
        seq_num = 6
    elif dataset in ('cifar2', 'old_cifar2'):
        seq_len = 4 if feature_size == '2x2' else 9
        # loader = load_cifar2 if dataset == 'cifar2' else load_old_cifar2
        # x_train, y_train, x_test, y_test = loader(n_train, n_test, seq_len=seq_len, classify_choice=classify_choice)
        if dataset == 'cifar2':
            x_train, y_train, x_test, y_test = load_cifar2(n_train, n_test, seq_len=seq_len, classify_choice=classify_choice)
        else:
            x_train, y_train, x_test, y_test = load_old_cifar2(n_train, n_test, seq_len=seq_len, classify_choice=classify_choice)
        seq_num = seq_len
    else:
        raise ValueError("Available datasets: 'mnist', 'cifar2', 'old_cifar2'")

    # Initialize model and training components
    model = QRNN(anc_q=3, n_qub_enc=1, seq_num=seq_num, D=2)
    optimizer = create_optimizer()
    opt_state = optimizer.init(model.params)
    train_step = make_train_step(model.circuit, optimizer)
    evaluate = make_evaluate(model.circuit)
    forward_pass = make_forward_pass(model.circuit)

    # Determine batch size
    batch_size = get_batch_size(n_train)

    # Training loop with loss and accuracy storage
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    top_error_train, top_error_test = [], []
    total_time = 0
    print(f"Training QRNN on {dataset} ({n_train} train, {n_test} test, {n_epochs} epochs, batch_size={batch_size})...")
    for epoch in range(n_epochs):
        start_epoch = time.time()
        key = jax.random.PRNGKey(epoch)
        train_loss_epoch, num_batches = 0.0, 0
        
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size, key):
            model.params, opt_state, batch_loss = train_step(model.params, opt_state, x_batch, y_batch)
            train_loss_epoch += batch_loss
            num_batches += 1
        
        train_loss_epoch /= num_batches
        train_loss_val, train_acc_val = evaluate(model.params, x_train, y_train)
        test_loss_val, test_acc_val = evaluate(model.params, x_test, y_test)
        train_losses.append(float(train_loss_epoch))
        test_losses.append(float(test_loss_val))
        train_accs.append(float(train_acc_val))
        test_accs.append(float(test_acc_val))
        top_error_train.append(1.0 - train_acc_val)
        top_error_test.append(1.0 - test_acc_val)
        total_time += time.time() - start_epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss_val:.4f}, Acc: {train_acc_val:.4f}")
            print(f"Test Loss: {test_loss_val:.4f}, Acc: {test_acc_val:.4f}")
            print(f"Approx 10 epochs time: {10*(time.time() - start_epoch):.2f} seconds")

    # Compute metrics
    top1_error_train = float(np.array(top_error_train).max())
    top1_error_test = float(np.array(top_error_test).max())
    top5_error_train = float(sorted(top_error_train, reverse=True)[4] if len(top_error_train) >= 5 else top1_error_train)
    top5_error_test = float(sorted(top_error_test, reverse=True)[4] if len(top_error_test) >= 5 else top1_error_test)
    gen_err = float(generalization_error(train_loss_val, test_loss_val))
    y_pred = (forward_pass(model.params, x_test) > 0.5).astype(int).flatten()
    precision, recall, f1 = map(float, (precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))

    print(f"Generalization Error: {gen_err:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")

    # Save or display confusion matrix
    plot_confusion_matrix(y_test, y_pred, dataset, classify_choice, main_folder, desc, n_train, log)

    # Save or print metrics
    if log:
        os.makedirs(f'{main_folder}/generalization_results', exist_ok=True)
        os.makedirs(f'{main_folder}/top_error_results', exist_ok=True)
        
        save_to_csv(pd.DataFrame({
            'name': [f'QRNN_{dataset}_ntrain{n_train}_batch{batch_size}_{desc}'],
            'train_losses': [float(train_loss_val)],
            'test_losses': [float(test_loss_val)],
            'train_accs': [float(train_acc_val)],
            'test_accs': [float(test_acc_val)],
            'generalization_error': [gen_err]
        }), f'{main_folder}/generalization_results/clean_results.csv', mode, header)
        
        save_to_csv(pd.DataFrame({
            'name': [f'{dataset}_ntrain{n_train}_{desc}'],
            'generalization_error': [gen_err],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1]
        }), f'{main_folder}/generalization_results/important_metrics.csv', mode, header)
        
        save_to_csv(pd.DataFrame({
            'name': [f'{dataset}_ntrain{n_train}_{desc}'],
            'top1_error_train': [top1_error_train],
            'top1_error_test': [top1_error_test],
            'top5_error_train': [top5_error_train],
            'top5_error_test': [top5_error_test]
        }), f'{main_folder}/top_error_results/top_error_results.csv', mode, header)
    else:
        print("Metrics not saved (log=False).")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print(f"Top 1 Error Train: {top1_error_train:.4f}, Test: {top1_error_test:.4f}")
        print(f"Top 5 Error Train: {top5_error_train:.4f}, Test: {top5_error_test:.4f}")
        print(f"Generalization Error: {gen_err:.4f}")

    # Adversarial evaluation
    adv_metrics = []
    clean_acc = None
    if evaluate_adv:
        attacks = [
            ('FGSM', lambda x, y: FGSM(forward_pass, model.params, x, y, eps=0.2)),
            ('PGD', lambda x, y: PGD(forward_pass, model.params, x, y, eps=0.2, alpha=0.02, steps=5)),
            ('APGD', lambda x, y: APGD(forward_pass, model.params, x, y, eps=0.2, alpha=0.01, steps=3)),
            ('MIM', lambda x, y: MIM(forward_pass, model.params, x, y, eps=0.2, alpha=0.02, steps=10)),
            ('SA', lambda x, y: SA(forward_pass, model.params, x, y, eps=0.2, steps=10, p=0.05))
        ]
        print("\nAdversarial Evaluation:")
        clean_loss, clean_acc = evaluate(model.params, x_test, y_test)
        print(f"Clean Test Loss: {clean_loss:.4f}, Acc: {clean_acc:.4f}")
        
        for attack_name, attack_fn in attacks:
            print(f"\n{attack_name} Attack:")
            x_adv = attack_fn(x_test, y_test)
            adv_loss, adv_acc = evaluate(model.params, x_adv, y_test)
            asr = 1.0 - adv_acc
            robustness_gap = clean_acc - adv_acc
            print(f"Adversarial Test Loss: {adv_loss:.4f}, Acc: {adv_acc:.4f}")
            print(f"Attack Success Rate: {asr:.4f}, Robustness Gap: {robustness_gap:.4f}")
            
            adv_metrics.append({
                'name': f'{dataset}_ntrain{n_train}_{desc}',
                'Attack': attack_name,
                'Clean_Loss': float(clean_loss),
                'Clean_Acc': float(clean_acc),
                'Adversarial_Loss': float(adv_loss),
                'Adversarial_Acc': float(adv_acc),
                'Attack_Success_Rate': float(asr),
                'Robustness_Gap': float(robustness_gap)
            })

        if log:
            os.makedirs(f'{main_folder}/adversarial_results', exist_ok=True)
            save_to_csv(pd.DataFrame(adv_metrics), f'{main_folder}/adversarial_results/metrics.csv', mode, header)

    return {
        'dataset_name': f'{dataset}_ntrain{n_train}_{desc}',
        'clean_acc': float(clean_acc) if clean_acc is not None else None,
        'adv_metrics': adv_metrics,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_error': gen_err
    } if log else None

def run_experiments(experiments: List[Dict], train_sizes: List[int],
                    test_sizes: List[int], n_reps: int = 5, desc: str = '',
                    clear_dir_auto: bool = False, log: bool = True) -> None:
    all_results = []
    for exp in experiments:
        train_loss_lsts_size = []
        test_loss_lsts_size = []
        train_acc_lsts_size = []
        test_acc_lsts_size = []
        gen_errors = []
        
        for idx, (n_train, n_test) in enumerate(zip(train_sizes, test_sizes)):
            t_losses = []
            te_losses = []
            tr_acc = []
            te_acc = []
            gen_errs = []
            clean_accs = []
            adv_metrics_all = {attack: [] for attack in ['FGSM', 'PGD', 'APGD', 'MIM', 'SA']}
            
            for rep in range(n_reps):
                print(f"Training for {exp['dataset']} with {n_train} train, {n_test} test, rep {rep + 1}/{n_reps}")
                # Set mode to 'w' for first rep of first training size, 'a' otherwise
                mode = exp['mode'] if idx == 0 and rep == 0 else 'a'
                result = train_and_evaluate_adversarial(
                    n_train=n_train,
                    n_test=n_test,
                    n_epochs=100,
                    dataset=exp['dataset'],
                    classify_choice=exp['classify_choice'],
                    desc=f"{desc + exp['desc']}_rep{rep+1}",
                    mode=mode,
                    log=log,
                    clear_dir_auto=clear_dir_auto and mode == 'w'
                )
                if result:
                    t_losses.append(result['train_losses'])
                    te_losses.append(result['test_losses'])
                    tr_acc.append(result['train_accs'])
                    te_acc.append(result['test_accs'])
                    gen_errs.append(result['generalization_error'])
                    if result['clean_acc'] is not None:
                        clean_accs.append(result['clean_acc'])
                    for adv_metric in result['adv_metrics']:
                        adv_metrics_all[adv_metric['Attack']].append({
                            'Clean_Loss': adv_metric['Clean_Loss'],
                            'Clean_Acc': adv_metric['Clean_Acc'],
                            'Adversarial_Loss': adv_metric['Adversarial_Loss'],
                            'Adversarial_Acc': adv_metric['Adversarial_Acc'],
                            'Attack_Success_Rate': adv_metric['Attack_Success_Rate'],
                            'Robustness_Gap': adv_metric['Robustness_Gap']
                        })
            
            # Compute mean metrics across repetitions
            if t_losses:
                mean_train_l = np.mean(t_losses, axis=0)
                mean_test_l = np.mean(te_losses, axis=0)
                mean_train_acc = np.mean(tr_acc, axis=0)
                mean_test_acc = np.mean(te_acc, axis=0)
                mean_gen_err = np.mean(gen_errs)
                
                train_loss_lsts_size.append(mean_train_l)
                test_loss_lsts_size.append(mean_test_l)
                train_acc_lsts_size.append(mean_train_acc)
                test_acc_lsts_size.append(mean_test_acc)
                gen_errors.append(mean_gen_err)
                
                # Compute mean adversarial metrics
                mean_adv_metrics = []
                for attack in adv_metrics_all:
                    if adv_metrics_all[attack]:
                        mean_metrics = {
                            'name': f"{exp['dataset']}_ntrain{n_train}_{desc + exp['desc']}",
                            'Attack': attack,
                            'Clean_Loss': float(np.mean([m['Clean_Loss'] for m in adv_metrics_all[attack]])),
                            'Clean_Acc': float(np.mean([m['Clean_Acc'] for m in adv_metrics_all[attack]])),
                            'Adversarial_Loss': float(np.mean([m['Adversarial_Loss'] for m in adv_metrics_all[attack]])),
                            'Adversarial_Acc': float(np.mean([m['Adversarial_Acc'] for m in adv_metrics_all[attack]])),
                            'Attack_Success_Rate': float(np.mean([m['Attack_Success_Rate'] for m in adv_metrics_all[attack]])),
                            'Robustness_Gap': float(np.mean([m['Robustness_Gap'] for m in adv_metrics_all[attack]]))
                        }
                        mean_adv_metrics.append(mean_metrics)
                
                # Append mean result to all_results
                mean_clean_acc = float(np.mean(clean_accs)) if clean_accs else None
                all_results.append({
                    'dataset_name': f"{exp['dataset']}_ntrain{n_train}_{desc + exp['desc']}",
                    'clean_acc': mean_clean_acc,
                    'adv_metrics': mean_adv_metrics,
                    'train_losses': mean_train_l.tolist(),
                    'test_losses': mean_test_l.tolist(),
                    'train_accs': mean_train_acc.tolist(),
                    'test_accs': mean_test_acc.tolist(),
                    'generalization_error': mean_gen_err
                })
                
                print(f"Mean metrics for {exp['dataset']} with {n_train} train: "
                      f"Train Loss: {mean_train_l[-1]:.4f}, Test Loss: {mean_test_l[-1]:.4f}, "
                      f"Train Acc: {mean_train_acc[-1]:.4f}, Test Acc: {mean_test_acc[-1]:.4f}, "
                      f"Gen Error: {mean_gen_err:.4f}")

        # Plot generalization errors for all training sizes
        if log and train_loss_lsts_size:
            gen_plotter2(
                train_size_list=train_sizes,
                train_loss_lsts_size=train_loss_lsts_size,
                test_loss_lsts_size=test_loss_lsts_size,
                train_acc_lsts_size=train_acc_lsts_size,
                test_acc_lsts_size=test_acc_lsts_size,
                generalization_errors=gen_errors,
                desc=f'{exp["dataset"]}_{exp["classify_choice"][0]}_{exp["classify_choice"][1]}_{desc + exp["desc"]}'
            )

    main_folder = 'QRNN_JAX'
    plot_multi_barplot(all_results, 'Accuracy', 'Clean and Adversarial Accuracy Across Datasets', 'Accuracy', main_folder, 'multi_barplot_clean_accuracy.png')
    plot_multi_barplot(all_results, 'Adversarial Accuracy', 'Adversarial Accuracy Across Datasets', 'Adversarial Accuracy', main_folder, 'multi_barplot_adversarial_accuracy.png')
    plot_multi_barplot(all_results, 'Attack Success Rate', 'Attack Success Rate Across Datasets', 'Attack Success Rate', main_folder, 'multi_barplot_attack_success_rate.png')

if __name__ == "__main__":
    train_sizes = [500, 1000, 2000, 5000, 10000]
    test_sizes = [i // 5 for i in train_sizes]
    n_reps = 5
    experiments = [
        {'dataset': 'cifar2', 'classify_choice': [0, 1], 'desc': '', 'mode': 'w'},
        {'dataset': 'cifar2', 'classify_choice': [1, 9], 'desc': '1_5', 'mode': 'a'},
        {'dataset': 'mnist', 'classify_choice': [1, 7], 'desc': '', 'mode': 'a'},
        {'dataset': 'mnist', 'classify_choice': [6, 9], 'desc': '6_9', 'mode': 'a'},
        {'dataset': 'mnist', 'classify_choice': [0, 8], 'desc': '0_8', 'mode': 'a'}
    ]
    run_experiments(experiments, train_sizes, test_sizes, n_reps=n_reps, clear_dir_auto=True, log=True)