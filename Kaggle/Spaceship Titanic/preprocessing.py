import featureengineering
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['CabinNum']
categorical_features_onehot = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
targer_label = ['Transported']

def categorical_Encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, features):
    '''
    
    '''
    dict_encoder = {}
    concat_data = pd.concat([train_df[features], test_df[features]])

    for col in concat_data.columns:
        print('Encoding', col, '...')
        encoder = LabelEncoder()
        encoder.fit(concat_data[col])
        dict_encoder[col] = encoder

        train_df[col + '_Enc'] = encoder.transform(train_df[col])
        test_df[col + '_Enc'] = encoder.transform(test_df[col])
    
    train_df = train_df.drop(columns=features, axis=1)
    test_df = test_df.drop(columns=features, axis=1)

    return train_df, test_df

def one_hot(df: pd.DataFrame, features):
    for col in features:
        tmp = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, tmp], axis=1)
    
    df = df.drop(columns=features)
    return df



def preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = featureengineering.feature_engineering(train_df)
    test_df = featureengineering.feature_engineering(test_df)

    train_df, test_df = categorical_Encoder(train_df, test_df, categorical_features)
    train_df = one_hot(train_df, categorical_features_onehot)
    test_df = one_hot(test_df, categorical_features_onehot)

    train_label = train_df.loc[:, 'Transported']
    train_data = train_df.drop('Transported', axis=1)
    train_label = train_label.astype(int)

    train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

    ss = StandardScaler()
    ss.fit(train_data)
    train_data = ss.transform(train_data)
    valid_data = ss.transform(valid_data)
    test_data = ss.transform(test_df)

    return train_data, valid_data, test_data, train_label, valid_label

if __name__ == '__main__':
    train_csv = pd.read_csv('./datasets/train.csv')
    test_csv = pd.read_csv('./datasets/test.csv')
    # print(train_csv.head())
    train_data, valid_data, test_data, train_label, valid_label = preprocessing(train_csv, test_csv)
    # print(test_data)
    # features = [fea for fea in train_df.columns]
    # print(features)