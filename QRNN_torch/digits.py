# from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import datasets
# digits = datasets.load_digits()

class DigitsLoader():
    def __init__(self, n_train = None, test_size = 0.2):
        self.digits = datasets.load_digits()
        self.test_size = test_size
        X = self.digits.data
        y = self.digits.target
        X, y = X[np.where((y == 0) | (y == 1))], y[np.where((y == 0) | (y == 1))]

        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = self.test_size, random_state = 42)
        if n_train is None:
            self.n_train = len(y_train)
        else:
            self.n_train = n_train
        X_train, y_train = X_train[:self.n_train], y_train[:self.n_train]
        X_train, X_test = X_train.reshape(-1,8,8), X_test.reshape(-1,8,8)
        X_train = torch.tensor(X_train).unsqueeze(1).float()
        X_test = torch.tensor(X_test).unsqueeze(1).float()

        pooled_train = F.adaptive_avg_pool2d(X_train, output_size=(3,3))
        pooled_test = F.adaptive_avg_pool2d(X_test, output_size=(3,3))

        train_seq = pooled_train.squeeze(1)
        test_seq = pooled_test.squeeze(1)

        _min, _max = train_seq.min(), train_seq.max()
        train_seq = ((train_seq - _min) / (_max - _min)) * torch.pi
        test_seq = ((test_seq - _min) / (_max - _min)) * torch.pi

        train = train_seq.reshape(-1, train_seq.shape[1] * train_seq.shape[2])
        test = test_seq.reshape(-1, test_seq.shape[1] * test_seq.shape[2])
        self.trainset = data_utils.TensorDataset(train, torch.tensor(y_train))
        self.testset = data_utils.TensorDataset(test, torch.tensor(y_test))
        self.length = X_train.shape[0]
        self.length_test = X_test.shape[0]
    def get_loaders(self, shuffle=True):
        batch_size = self.length
        test_batch_size = self.length_test
        trainloader = data_utils.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
        testloader = data_utils.DataLoader(self.testset, batch_size=test_batch_size, shuffle=shuffle)
        return trainloader, testloader
    def quick_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=12, figsize=(3, 1))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(self.digits.data[i].reshape((8, 8)), cmap="gray")
            ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


