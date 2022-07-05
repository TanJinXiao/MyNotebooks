
import torch

from model import ResNet18
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt

dataNum, correctNum = 0, 0

def acc_calculation(pred, y):
    global dataNum, correctNum
    for idx in range(len(y)):
        dataNum += 1
        if pred[idx] == y[idx]:
            correctNum += 1

def train(model: ResNet18, optimizer: AdamW, scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, loss_func: CrossEntropyLoss, train_iter: DataLoader, num_epoch, learning_rate, device='cpu'):
    best_acc, best_epoch, best_loss = 0, 0, 1.0
    train_acc, train_loss = [], []
    model = model.to(device)
    print("===========training=========")
    for epoch in range(num_epoch):
        acc, lossNum = 0, 0
        dataNum, correctNum = 0, 0
        length = 0
        for step, (x, y) in enumerate(train_iter):
            # print(torch.typename(x))
            x, y = x.to(device), y.to(device)
            length += x.shape[0]
            optimizer.zero_grad()
            output = model(x)
            acc += (torch.argmax(output, dim=1) == y).to('cpu').sum().float().item()
            loss = loss_func(output, y)
            lossNum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss.append(lossNum / length)
        train_acc.append(acc / length)
        if(loss < best_loss):
            best_loss = loss
            best_epoch = epoch+1
            torch.save(model, 'MINIST_model.pkl')
        print("epoch{}: acc: {}, loss: {}".format(epoch, acc / length, lossNum / length))
    print("=============================")         
    print("best acc: {}, best epoch: {}".format(best_acc, best_epoch))
    
    plt.plot(range(1, num_epoch+1), train_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
