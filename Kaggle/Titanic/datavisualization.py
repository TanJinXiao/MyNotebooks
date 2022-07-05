from traceback import print_tb
import torch

import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import missingno

def main():
    train_csv = pd.read_csv('./datasets/train.csv')
    test_csv = pd.read_csv('./datasets/test.csv')
    gender_csv = pd.read_csv('./datasets/gender_submission.csv')
    print('===============original data===============')
    print('-----train_csv.head:----- \n', train_csv.head())
    print('-----test_csv.head:----- \n', test_csv.head())
    print('-----gender_csv.head:----- \n',gender_csv.head())

    print('===============Data Overview===============')
    print('train_csv.shape: ', train_csv.shape)
    print('test_csv.shape: ',test_csv.shape)
    print('gerder_csv.shape: ', gender_csv.shape)
    print('train_csv.columns: ', train_csv.columns)
    print('test_csv.columns: ', test_csv.columns)

    print('----------train_csv.info:---------- \n', train_csv.info())
    print('----------trian_csv NULL sum:---------- \n',train_csv.isnull().sum())
    

if __name__ == '__main__':
    main()
