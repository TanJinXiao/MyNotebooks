import torch
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import missingno

def main():
    train_csv = pd.read_csv('./datasets/train.csv')
    test_csv = pd.read_csv('./datasets/test.csv')
    sub_csv = pd.read_csv('./datasets/sample_submission.csv')

    print('==========Original Data========== \n')

    print('-----training set info-----')
    # print('training set head(): \n', train_csv.head())
    '''
    PassengerId HomePlanet CryoSleep  Cabin  Destination   Age    VIP  RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name  Transported
    0     0001_01     Europa     False  B/0/P  TRAPPIST-1e  39.0  False          0.0        0.0           0.0     0.0     0.0    Maham Ofracculy        False
    1     0002_01      Earth     False  F/0/S  TRAPPIST-1e  24.0  False        109.0        9.0          25.0   549.0    44.0       Juanna Vines         True
    2     0003_01     Europa     False  A/0/S  TRAPPIST-1e  58.0   True         43.0     3576.0           0.0  6715.0    49.0      Altark Susent        False
    3     0003_02     Europa     False  A/0/S  TRAPPIST-1e  33.0  False          0.0     1283.0         371.0  3329.0   193.0       Solam Susent        False
    4     0004_01      Earth     False  F/1/S  TRAPPIST-1e  16.0  False        303.0       70.0         151.0   565.0     2.0  Willy Santantines         True
    '''
    # print('\n training set shape: ', train_csv.shape) # (8693, 14)
    # print('\n training set info: \n', train_csv.info())
    '''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8693 entries, 0 to 8692
    Data columns (total 14 columns):
    #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
    0   PassengerId   8693 non-null   object 
    1   HomePlanet    8492 non-null   object 
    2   CryoSleep     8476 non-null   object 
    3   Cabin         8494 non-null   object 
    4   Destination   8511 non-null   object 
    5   Age           8514 non-null   float64
    6   VIP           8490 non-null   object 
    7   RoomService   8512 non-null   float64
    8   FoodCourt     8510 non-null   float64
    9   ShoppingMall  8485 non-null   float64
    10  Spa           8510 non-null   float64
    11  VRDeck        8505 non-null   float64
    12  Name          8493 non-null   object 
    13  Transported   8693 non-null   bool   
    dtypes: bool(1), float64(6), object(7)
    memory usage: 891.5+ KB
    '''
    # print('\n training set null count: \n', train_csv.isnull().sum())
    '''
    PassengerId       0
    HomePlanet      201
    CryoSleep       217
    Cabin           199
    Destination     182
    Age             179
    VIP             203
    RoomService     181
    FoodCourt       183
    ShoppingMall    208
    Spa             183
    VRDeck          188
    Name            200
    Transported       0
    dtype: int64
    '''
    print('----------------------------')

    print('-----test set info-----')
    print('test set head(): \n', test_csv.head())
    '''
    PassengerId	HomePlanet	CryoSleep	Cabin	Destination	Age	    VIP	    RoomService	  FoodCourt	ShoppingMall	Spa	    VRDeck	Name
0	0013_01	    Earth	    True	    G/3/S	TRAPPIST-1e	27.0	False	0.0	          0.0	    0.0	            0.0	    0.0	    Nelly Carsoning
1	0018_01	    Earth	    False	    F/4/S	TRAPPIST-1e	19.0	False	0.0	          9.0	    0.0	            2823.0	0.0	    Lerome Peckers
2	0019_01	    Europa	    True	    C/0/S	55 Cancri e	31.0	False	0.0	          0.0	    0.0	            0.0	    0.0	    Sabih Unhearfus
3	0021_01	    Europa	    False	    C/1/S	TRAPPIST-1e	38.0	False	0.0	          6652.0    0.0	            181.0	585.0	Meratz Caltilter
4	0023_01	    Earth	    False	    F/5/S	TRAPPIST-1e	20.0	False	10.0	      0.0	    635.0	        0.0	    0.0	    Brence Harperez
    '''
    # print('\n test set shape: ', test_csv.shape) # (4277, 13)
    # print('\n test set info: \n', test_csv.info())
    '''<class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4277 entries, 0 to 4276
    Data columns (total 13 columns):
    #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
    0   PassengerId   4277 non-null   object 
    1   HomePlanet    4190 non-null   object 
    2   CryoSleep     4184 non-null   object 
    3   Cabin         4177 non-null   object 
    4   Destination   4185 non-null   object 
    5   Age           4186 non-null   float64
    6   VIP           4184 non-null   object 
    7   RoomService   4195 non-null   float64
    8   FoodCourt     4171 non-null   float64
    9   ShoppingMall  4179 non-null   float64
    10  Spa           4176 non-null   float64
    11  VRDeck        4197 non-null   float64
    12  Name          4183 non-null   object 
    dtypes: float64(6), object(7)'''
    # print('\n test set null count: \n', test_csv.isnull().sum())
    '''PassengerId       0
    HomePlanet       87
    CryoSleep        93
    Cabin           100
    Destination      92
    Age              91
    VIP              93
    RoomService      82
    FoodCourt       106
    ShoppingMall     98
    Spa             101
    VRDeck           80
    Name             94
    dtype: int64'''
    print('----------------------------')

    

if __name__ == '__main__':
    main()