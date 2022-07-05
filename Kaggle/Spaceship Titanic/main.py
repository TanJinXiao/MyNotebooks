from pyexpat import model
import torch
import pandas as pd
import numpy as np

import featureengineering
import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

train_csv = pd.read_csv('./datasets/train.csv')
test_csv = pd.read_csv('./datasets/test.csv')
sub_csv = pd.read_csv('./datasets/sample_submission.csv')

train_data, valid_data, test_data, train_label, valid_label = preprocessing.preprocessing(train_csv, test_csv)

def model_prediction(model):
    cv = cross_val_score(model, train_data, train_label, cv=5)
    print('Cross Validation Score: {}'.format(cv))
    print('Mean Cross Validation Score: {}'.format(cv.mean()))

    model.fit(train_data, train_label)
    pred = model.predict(valid_data)
    accuracy = round(accuracy_score(valid_label, pred)*100, 2)
    print('Model:{} Accuracy: {}%'.format(type(model).__name__,accuracy))
    model_pred = model.predict(test_data)
    return model_pred

def model_submission(model_pred):
    prediction = {'PassengerId': sub_csv.PassengerId, 'Transported': model_pred.astype('bool')}
    submission = pd.DataFrame(data=prediction)
    print(submission['Transported'].value_counts())
    return submission

def train_process(model):
    pred = model_prediction(model)
    submission = model_submission(pred)
    submission.to_csv('{}_submission.csv'.format(type(model).__name__), index=False)

def main():
    
    train_process(LogisticRegression())
    
    train_process(SGDClassifier())

    # train_process(LinearSVC())

    train_process(KNeighborsClassifier(n_neighbors=13))

    train_process(DecisionTreeClassifier(random_state=1))

    train_process(RandomForestClassifier(bootstrap=True, max_depth=7, n_estimators=550, random_state=42))
    

if __name__ == '__main__':
    # print(test_csv.head())
    main()