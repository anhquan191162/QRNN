import torch
import torch.nn as nn
# from ipywidgets import interactive
from tqdm import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from joblib import load, dump
# import progressbar
def trainer(model, epochs, lr, trainloader, testloader, eval=False, show=True):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    test_loss_lsts = []
    train_loss_lsts = []
    train_acc = []
    test_accs = []
    for epoch in range(epochs):
        acc = 0
        train_loss_lst = []
        pbar = tqdm(trainloader, leave=True, desc=f"Epoch {epoch}")
        for batch, (feature, label) in enumerate(pbar):
            optimizer.zero_grad()
            feature, label = feature.to(device), label.to(device)

            preds = model(feature.float())
            loss = criterion(preds, label.float())

            preds, label = preds.ravel(), label.ravel()
            acc += binary_accuracy(preds, label)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch / len(trainloader))
            train_loss_lst.append(loss.cpu().item())

            if show:  # Only update progress bar if 'show' is True
                pbar.set_description(f"Epoch {epoch} | Batch {batch+1} / {len(trainloader)} | Loss {loss.item():.4f} | Accuracy {acc/(batch+1):.4f}")

        acc = acc / (batch + 1)
        train_acc.append(acc)
        train_loss_lsts.append(sum(train_loss_lst)/len(train_loss_lst))
        print(f'Epoch {epoch} | Training Accuracy: {acc:.4f}')

        if eval is True:
            model.eval()
            with torch.no_grad():
                test_acc = 0
                test_loss_lst = []
                tbar = tqdm(testloader, leave=True, desc=f"Eval_epoch {epoch}")
                for batch, (feature, label) in enumerate(tbar):
                    feature, label = feature.to(device), label.to(device)
                    preds = model(feature.float())
                    loss = criterion(preds, label.float())
                    preds, label = preds.ravel(), label.ravel()
                    test_acc += binary_accuracy(preds, label)
                    test_loss_lst.append(loss.cpu().item())
                    tbar.set_description(f"Eval_Epoch {epoch} | Batch {batch+1} / {len(testloader)} | Loss {loss.item():.4f} | Accuracy {test_acc/(batch+1):.4f}")
                test_acc = test_acc / (batch + 1)
                test_loss_lsts.append(sum(test_loss_lst)/len(test_loss_lst))
                print(f'Epoch {epoch} | Eval Accuracy: {test_acc:.4f}')
                test_accs.append(test_acc)

    return model, train_loss_lsts, train_acc, test_loss_lsts, test_accs


def tester(model, testloader, show=False):
    preds = []
    labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    if show:
        pbar = tqdm(testloader, leave=True, desc="Testing")
    else:
        pbar = testloader

    for batch, (feature, label) in enumerate(pbar):
        feature, label = feature.to(device), label.to(device)
        with torch.no_grad():
            pred = model(feature.float())
            preds.append(pred.cpu())
            labels.append(label.cpu())

        if show:
            # Calculate accuracy for this batch
            rounded_preds = ((torch.round(torch.sign(pred - 0.5)) + 1) // 2)
            acc = 1 - (torch.abs(rounded_preds - label).mean())
            pbar.set_description(
                f"Testing | Batch {batch+1}/{len(testloader)} | Accuracy {acc.item():.4f}"
            )

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    rounded_preds = ((torch.round(torch.sign(preds - 0.5)) + 1) // 2)
    acc = 1 - (torch.abs(rounded_preds - labels).mean())

    return rounded_preds, acc, labels



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = (torch.round(torch.sign(preds-0.5))+1)//2
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


