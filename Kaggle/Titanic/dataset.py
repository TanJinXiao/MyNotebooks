import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

train_csv_path = './datasets/train.csv'
test_csv_path = './datasets/test.csv'

def my_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.shape)

if __name__ == '__main__':
    my_dataset(train_csv_path)