import pandas as pd
from pyparsing import col

import torch

'''
	PassengerId	HomePlanet	CryoSleep	Cabin	Destination	Age	    VIP	    RoomService	FoodCourt	ShoppingMall	Spa	    VRDeck	Name	            Transported
0	0001_01	    Europa  	False   	B/0/P	TRAPPIST-1e	39.0	False	0.0	        0.0	        0.0	            0.0	    0.0	    Maham Ofracculy	    False
1	0002_01	    Earth	    False	    F/0/S	TRAPPIST-1e	24.0	False	109.0	    9.0	2       5.0	            549.0	44.0	Juanna Vines	    True
2	0003_01	    Europa  	False   	A/0/S	TRAPPIST-1e	58.0	True	43.0	    3576.0	    0.0	            6715.0	49.0	Altark Susent	    False
3	0003_02	    Europa  	False   	A/0/S	TRAPPIST-1e	33.0	False	0.0	        1283.0	    371.0	        3329.0	193.0	Solam Susent	    False
4	0004_01	    Earth	    False	    F/1/S	TRAPPIST-1e	16.0	False	303.0	    70.0	    151.0	        565.0	2.0	    Willy Santantines	True'''

def fill_billed_nans(df: pd.DataFrame, num=0):
    '''
    填充花费，可以选择以均值或0填充,如果是使用均值填充，可以直接跳过billed执行全部的fillna
    '''
    df['RoomService'].fillna(num, inplace=True)
    df['FoodCourt'].fillna(num, inplace=True)
    df['ShoppingMall'].fillna(num, inplace=True)
    df['Spa'].fillna(num, inplace=True)
    df['VRDeck'].fillna(num, inplace=True)
    return df

def fill_all_nans(df: pd.DataFrame):
    '''
    填充所有nans
    使用均值或众数进行填充
    '''
    numerices = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric_tmp = df.select_dtypes(include=numerices)
    category_tmp = df.select_dtypes(exclude=numerices)

    print('numeric cols:')
    for col in numeric_tmp:
        print(col)
        df[col].fillna(df[col].mean(), inplace=True)
    
    print('category cols:')
    for col in category_tmp:
        print(col)
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def split_PassengerId(df: pd.DataFrame):
    '''
    分割PassengerId,其实我认为group对结果的影响应该很小
    '''
    df['PassengerGroup'] = df['PassengerId'].str.split('_', expand=True)[0]

    df.drop('PassengerId', axis=1, inplace=True)
    return df

def split_Cabin(df: pd.DataFrame):
    '''
    Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
    '''
    df['CabinDeck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['CabinNum'] = df['Cabin'].str.split('/', expand=True)[1]
    df['CabinSide'] = df['Cabin'].str.split('/', expand=True)[2]
    
    df.drop('Cabin', axis=1, inplace=True)
    return df

def feature_engineering(df: pd.DataFrame):
    '''
    做特征处理
    其中删除了我认为无关的特征，包括Name, PassengerId中Group内的ID
    '''
    df = fill_billed_nans(df)
    df = fill_all_nans(df)
    df = split_PassengerId(df)
    df = split_Cabin(df)
    df.drop(columns=['Name'], inplace=True)
    return df