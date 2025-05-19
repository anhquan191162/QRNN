import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from helper import trainer, tester
from digits import DigitsLoader
from MODEL import *
def run_it(train_sizes, n_reps, epochs = 100):
    train_loss_lsts_size = []
    eval_loss_lsts_size = []
    train_acc_size = []
    test_acc_size = []
    gen_error = []
    for idx, size in enumerate(train_sizes):
        t_losses = []
        eval_l = []
        tr_acc = []
        te_acc = []
        for t in range(n_reps):
            trainloader, testloader = DigitsLoader(n_train=size).get_loaders()
            main_model = qrnn_lightning(3,1,4,2)
            model, train_loss, train_accuracies, test_loss, test_accuracies = trainer(
                                                                                        main_model, lr=0.01, epochs=epochs, trainloader=trainloader, 
                                                                                        testloader=testloader, eval=True,show=False
                                                                                    )
            t_losses.append(train_loss)
            eval_l.append(test_loss)
            tr_acc.append(train_accuracies)
            te_acc.append(test_accuracies)
            rounded_preds, acc, labels = tester(model=model, testloader=testloader)
            print('{} accuracy at size {} and {} reps: '.format(acc,size,t+1))
        mean_train_l, mean_eval_l, mean_train_acc, mean_eval_acc = (np.mean(t_losses,axis=0), np.mean(eval_l,axis=0), 
                                                                    np.mean(tr_acc,axis=0), np.mean(te_acc,axis=0)
                                                                   )
        train_loss_lsts_size.append(mean_train_l)
        eval_loss_lsts_size.append(mean_eval_l)
        train_acc_size.append(mean_train_acc)
        test_acc_size.append(mean_eval_acc)
        gen_err = mean_eval_l[-1] - mean_train_l[-1]
        gen_error.append(gen_err)
    return train_loss_lsts_size, eval_loss_lsts_size, train_acc_size, test_acc_size, gen_error

def gen_plotter2(train_size_list, train_loss_lsts_size, test_loss_lsts_size,
                train_acc_lsts_size, test_acc_lsts_size, generalization_errors):

    r = np.linspace(0,99,100)
    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))
    lines = ["o-", "x--"]

    # === Loss and Accuracy Plots (Smoothed) ===
    for j, n_train in enumerate(train_size_list):
        labels = [fr"$N={n_train}$", None]

        sm_train_loss = smooth_curve(train_loss_lsts_size[j])
        sm_test_loss = smooth_curve(test_loss_lsts_size[j])
        sm_train_acc = smooth_curve(train_acc_lsts_size[j])
        sm_test_acc = smooth_curve(test_acc_lsts_size[j])

        axes[0].plot(r, sm_train_loss, lines[0], label=labels[0], markevery=20,
                     color=colors[j], alpha=0.8)
        axes[0].plot(r, sm_test_loss, lines[1], label=labels[1], markevery=20,
                     color=colors[j], alpha=0.8)

        axes[2].plot(r, sm_train_acc, lines[0], label=labels[0], markevery=20,
                     color=colors[j], alpha=0.8)
        axes[2].plot(r, sm_test_acc, lines[1], label=labels[1], markevery=20,
                     color=colors[j], alpha=0.8)

    axes[1].plot(train_size_list, np.array(generalization_errors) + 1e-6, "o-", label=r"$gen(\alpha)$", color='black')
    axes[1].set_xscale('log')
    axes[1].set_xticks(train_size_list)
    axes[1].set_xticklabels(train_size_list)
    axes[1].set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_yscale('log', base=2)
    axes[0].set_title('Train and Test Losses', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[2].set_title('Train and Test Accuracies', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0.45, 1.05)

    legend_elements = [
        mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_size_list)
    ] + [
        mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
        mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
    ]
    axes[0].legend(handles=legend_elements, ncol=3)
    axes[2].legend(handles=legend_elements, ncol=3)

    plt.tight_layout()
    plt.show() 


def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed


train_sizes = [2,5,10,20,40,80]
reps = 10
train_loss_lsts_size, eval_loss_lsts_size, train_acc_size, test_acc_size, gen_error = run_it(train_sizes, n_reps=reps)
# gen_plotter2(train_sizes,train_loss_lsts_size, eval_loss_lsts_size, train_acc_size, test_acc_size, gen_error)
