import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from train import train


def train_dataset(train_df):
    train_label = train_df['label'].values
    train_data = train_df.drop(['label'], axis=1).values

    train_label = np.eye(10)[train_label]

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).float()
    train_data[:784] /= 255.0
    train_data = train_data.reshape(-1, 1, 28, 28)

    # print(train_data.shape)
    # print(train_label.shape)
    return TensorDataset(train_data, train_label)

def test_dataset(test_df: pd.DataFrame):
    test_data = torch.from_numpy(test_df.to_numpy()).float()
    test_data /= 255.0
    test_data = test_data.reshape(-1, 1, 28, 28)
    test_label = torch.zeros(test_data.size()[0])
    return TensorDataset(test_data, test_label)


if __name__ == '__main__':
    train_df = pd.read_csv('./datasets/train.csv')
    train_data = train_dataset(train_df)
    for _, (example_data, example_label) in enumerate(train_data):
        print(example_data.shape)   
        print(example_label)