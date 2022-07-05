import train
import model
import dataset
import predict

import torch
import pandas as pd
from torch.utils.data import DataLoader



if __name__ == '__main__':
    train_df = pd.read_csv('./datasets/train.csv')
    test_df = pd.read_csv('./datasets/test.csv')
    # val_df = train_df.iloc[0.9 * len(train_df):]
    # train_df = train_df.iloc[:0.9 * len(train_df)]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    num_epoch = 20

    train_iter = DataLoader(dataset.train_dataset(train_df), 100)
    # val_iter = DataLoader(dataset.train_dataset(val_df), 100)
    test_iter = DataLoader(dataset.test_dataset(test_df), 100)

    # model = model.ResNet18(in_channels=1, num_classes=10)
    model = torch.load('./MINIST_model.pkl')
    # 
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)

    # print(test_iter)
    train.train(model, optimizer, scheduler, loss_func, train_iter, num_epoch, learning_rate, device)
    predict.predict(model, test_iter, device)
