from qrn_jax import *
from gen_plot import *
def run_experiments(train_sizes, test_sizes, n_reps, n_epochs, dataset = 'digits', classify_choice = [0,1]):
    print(f'Classification choice: {classify_choice}')
    print(f"Running experiments for {len(train_sizes)} sizes and {n_reps} reps.")
    start = time.time()
    train_loss_lsts_size = []
    test_loss_lsts_size = []
    train_acc_size = []
    test_acc_size = []
    gen_error = []

    for i in range(len(train_sizes)):
        t_losses = []
        te_losses = []
        tr_acc = []
        te_acc = []

        for _ in range(n_reps):
            print(f"Training for {train_sizes[i]} and testing for {test_sizes[i]} at rep {_ + 1}/{n_reps}")
            train_losses, test_losses, train_accs, test_accs = train_qrnn(train_sizes[i], test_sizes[i], n_epochs, encoding_type='angle', dataset = dataset, classify_choice = classify_choice, show = False)
            t_losses.append(train_losses)
            te_losses.append(test_losses)
            tr_acc.append(train_accs)
            te_acc.append(test_accs)    
        mean_train_l = np.mean(t_losses, axis=0)
        mean_test_l = np.mean(te_losses, axis=0)
        mean_train_acc = np.mean(tr_acc, axis=0)
        mean_test_acc = np.mean(te_acc, axis=0)
        print(f"Mean train loss: {mean_train_l[-1]}, Mean test loss: {mean_test_l[-1]}, Mean train acc: {mean_train_acc[-1]}, Mean test acc: {mean_test_acc[-1]}")
        train_loss_lsts_size.append(mean_train_l)
        test_loss_lsts_size.append(mean_test_l)
        train_acc_size.append(mean_train_acc)
        test_acc_size.append(mean_test_acc)
        gen_err = (mean_test_l[-1]) - (mean_train_l[-1])
        gen_error.append(gen_err)

    end = time.time()
    print(f"Total time taken: {end - start} seconds")

    return train_loss_lsts_size, test_loss_lsts_size, train_acc_size, test_acc_size, gen_error


# Example usage
if __name__ == "__main__":
    train_sizes = [500, 1000, 2000, 5000]
    test_sizes = [i//5 for i in train_sizes]
    n_reps = 5 #set to 100 for final experiments
    n_epochs = 100
    dataset = 'MNIST'.lower()
    if dataset== 'mnist':
        classify_choice = [1,7]
    elif dataset == 'cifar2':
        classify_choice = [0,1]
    elif dataset == 'digits':
        classify_choice = [0,1]

    train_loss_lsts_size, test_loss_lsts_size, train_acc_size, test_acc_size, gen_error = run_experiments(train_sizes, test_sizes, n_reps, n_epochs, dataset = dataset, classify_choice = classify_choice)
    gen_plotter2(train_sizes,train_loss_lsts_size, test_loss_lsts_size, train_acc_size, test_acc_size, gen_error, desc = f'{dataset}_{classify_choice[0]}_{classify_choice[1]}')