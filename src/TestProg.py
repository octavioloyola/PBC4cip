import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from core.PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary, WriteClassificationResults
from core.FileManipulation import WriteResultsCSV, returnX_y, get_dataframe_from_arff, GetFromFile
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMinerWithoutFiltering
from core.PatternFilter import MaximalPatternsGlobalFilter
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.Evaluation import obtainAUCMulticlass
from core.Helpers import ArgMax, convert_to_ndarray, get_col_dist, get_idx_val
from core.Dataset import Dataset, FileDataset, PandasDataset
from datetime import datetime

def CheckSuffix(file, suffix):
    if not suffix or len(suffix) == 0:
        return True
    if not suffix in file or len(suffix) >= len(file):
        return False
    filename, file_extension = os.path.splitext(file)
    return filename[(len(suffix)*-1):] == suffix


def GetFilesFromDirectory(directory):
    print(directory)
    files = []
    if os.path.isdir(directory):
        for r, d, f in os.walk(directory):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    else:
        raise Exception(f"Directory {directory} is not valid.")


def import_data(trainFile, testFile):
    train = pd.read_csv(trainFile, sep= ',') 
    test = pd.read_csv(testFile, sep= ',')

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
        confusion = [[0]*2 for i in range(numClasses)]
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
    #print(f"Test Scores: {y_test_scores}")
    
    y_pred = classifier.predict(y_test_scores)
    confusion, acc, auc = score(y_pred, y_test)

    
    print(f"Patterns Found: ")
    for pattern in patterns:
        print(f"{pattern}")

    print(f"\nConfusion Matrix:")
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"\n\nacc: {acc} , auc: {auc} , numPatterns: {len(patterns)}")
    

def Execute(args):
    testing_files = []
    training_files = []

    if args.training_files:
        training_files = args.training_files

    if args.training_directory:
        training_files = list(set().union(training_files, list(filter(
            lambda file: CheckSuffix(file, args.training_file_suffix), GetFilesFromDirectory(args.training_directory)))))

    if args.input_files:
        testing_files = args.input_files

    if args.input_directory:
        testing_files = list(set().union(testing_files, list(filter(
            lambda file: CheckSuffix(file, args.test_file_suffix), GetFilesFromDirectory(args.input_directory)))))

    training_files.sort()
    testing_files.sort()

    print(
        f"Training files detected (*{args.training_file_suffix}.[arff|dat]): {len(training_files)}")
    print(
        f"Testing files detected (*{args.test_file_suffix}.[arff|dat]): {len(testing_files)}")

    print("===============================================================================")
    tra = trange(len(training_files), desc='Training and Testing Files...', leave=True, unit="dataset")

    now = datetime.now()
    resultsId = now.strftime("%Y%m%d%H%M%S")
    for f in tra:
        tra.set_description(f"Working from {training_files[f]}")
        test_PBC4cip(training_files[f], testing_files[f] )



if __name__ == '__main__':

    defaultDataDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "data", "python")
    defaultOutputDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "output")

    defaultTrainingFiles = list()
    defaultTestingFiles = list()
    parser = argparse.ArgumentParser(
        description="Process class imbalanced datasets using PBC4cip.")

    parser.add_argument("--training-files",
                        type=str,
                        metavar="<*.dat/*.arff>",
                        nargs="+",
                        help="a file or files that are going to be used to train the classifier")

    parser.add_argument("--training-directory",
                        type=str,
                        metavar="'"+defaultDataDir+"'",
                        help="the directory with files to be used to train the classifier")

    parser.add_argument("--input-files",
                        type=str,
                        metavar="<*.dat/*.arff>",
                        nargs="+",
                        help="a file or files to be classified")
    
    parser.add_argument("--input-directory",
                        type=str,
                        metavar="'"+defaultDataDir+"'",
                        help="the directory with files to be classified")
    
    parser.add_argument("--output-directory",
                        type=str,
                        metavar="'"+defaultOutputDir+"'",
                        default=defaultOutputDir,
                        help="the output directory for the patterns")
    
    parser.add_argument("--test-file-suffix",
                        type=str,
                        metavar="'tst'",
                        default="tst",
                        help="states which suffix will indicate the test files")

    parser.add_argument("--training-file-suffix",
                        type=str,
                        metavar="'tra'",
                        default="tra",
                        help="states which suffix will indicate the training files")

    args = parser.parse_args()
    if not args.training_files and not args.training_directory and not args.input_files and not args.input_directory:
        parser.print_help()
    else:
        Execute(args)
