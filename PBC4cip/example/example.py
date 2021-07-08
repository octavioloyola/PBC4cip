import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from core.PBC4cip import PBC4cip
from core.Evaluation import obtainAUCMulticlass
from core.Helpers import get_col_dist, get_idx_val

def import_data(trainFile, testFile):
    train = pd.read_csv(trainFile) 
    test = pd.read_csv(testFile)
    return train, test

def split_data(train, test):
    X_train = train.iloc[:,  0:train.shape[1]-1]
    y_train =  train.iloc[:, train.shape[1]-1 : train.shape[1]]

    X_test = test.iloc[:,  0:test.shape[1]-1]
    y_test =  test.iloc[:, test.shape[1]-1 : test.shape[1]]

    return X_train, y_train, X_test, y_test

def score(predicted, y):
        y_class_dist = get_col_dist(y[f'{y.columns[0]}'])
        real = list(map(lambda instance: get_idx_val(y_class_dist, instance), y[f'{y.columns[0]}']))
        numClasses = len(y_class_dist)
        confusion = [[0]* numClasses for i in range(numClasses)]
        classified_as = 0
        error_count = 0

        for i in range(len(real)):
            if real[i] != predicted[i]:
                error_count = error_count + 1
            confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

        acc = 100.0 * (len(real) - error_count) / len(real)
        auc = obtainAUCMulticlass(confusion, numClasses)

        return confusion, acc, auc


def test_PBC4cip(trainFile, testFile):
    train, test = import_data(trainFile, testFile)
    X_train, y_train, X_test, y_test = split_data(train, test)
    
    classifier = PBC4cip()
    patterns = classifier.fit(X_train, y_train)

    y_test_scores = classifier.score_samples(X_test)
    print("Test Scores:")
    for i, test_score in enumerate(y_test_scores):
        print(f"{i}: {test_score}")
    
    y_pred = classifier.predict(X_test)
    confusion, acc, auc = score(y_pred, y_test)

    
    print(f"\nPatterns Found:")
    for pattern in patterns:
        print(f"{pattern}")

    print(f"\nConfusion Matrix:")
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"\n\nacc: {acc} , auc: {auc} , numPatterns: {len(patterns)}")
    
if __name__ == '__main__':
    current_location = os.path.dirname(os.path.abspath(__file__))
    trainFile = current_location + '\\train.csv'
    testFile = current_location + '\\test.csv'
    test_PBC4cip(trainFile, testFile)