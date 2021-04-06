import os
import argparse
import math
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from core.PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary
from core.FileManipulation import WriteResultsCSV, returnX_y, get_dataframe_from_arff, GetFromFile, convert_dat_to_csv
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMinerWithoutFiltering
from core.PatternFilter import MaximalPatternsGlobalFilter
from core.DistributionEvaluatorHelper import get_distribution_evaluator
from core.RandomSampler import SampleWithoutRepetition
from core.Evaluation import obtainAUCMulticlass
from core.SupervisedClassifier import DecisionTreeClassifier

from core.Helpers import ArgMax, convert_to_ndarray, get_col_dist, get_idx_val
from core.Dataset import Dataset, FileDataset, PandasDataset
from core.ResultsAnalyzer import show_results, wilcoxon, order_results
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

def run_C45(trainFile, outputDirectory, testFile, resultsId, distribution_evaluator, evaluationFunctionDir ):
    with open(evaluationFunctionDir, "r") as f:
        eval_functions = f.readlines()
        eval_functions = [line.replace("\n", "") for line in eval_functions]
   
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
    file_dataset = FileDataset(trainFile)
    dt_builder = DecisionTreeBuilder(file_dataset, X_train, y_train)
    if distribution_evaluator == 'combiner':
        dt_builder.distributionEvaluator = get_distribution_evaluator(distribution_evaluator)(eval_functions)
    else:
        dt_builder.distributionEvaluator = get_distribution_evaluator(distribution_evaluator)
    dt_builder.FeatureCount = int(math.log(len(file_dataset.Attributes), 2) + 1)
    dt_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
    dt = dt_builder.Build()
    dt_classifier = DecisionTreeClassifier(dt)
    y_scores = []
    for instance in X_test:
        inst_classify = dt_classifier.Classify(instance)
        #print(f"x_classify: {inst_classify}" )
        y_scores.append(inst_classify)
    
    y_pred = [ArgMax(instance) for instance in y_scores]
    #print(f"y_pred: {y_pred}")
    confusion, acc, auc = score_txtfile(y_pred, y_test, file_dataset)

    WriteResultsCSV(confusion, acc, auc, 0, testFile, outputDirectory, resultsId, "Not applicable", distribution_evaluator,
    functions_to_combine=eval_functions )
    show_results(confusion, acc, auc, 0)

def run_C45_multiple(trainFile, outputDirectory, testFile, resultsId, evaluationFunctionDir, eval_function_list):
    with open(eval_function_list, "r") as f:
        eval_functions = f.readlines()
        eval_functions = [line.replace("\n", "") for line in eval_functions]
    
    for func in eval_functions:
        run_C45(trainFile, outputDirectory, testFile, resultsId, func, evaluationFunctionDir)

def run_C45_combinations(trainFile, outputDirectory, testFile, resultsId, distribution_evaluator, evaluationFunctionDir, combination_size):
    if distribution_evaluator != 'combiner':
        raise Exception(f"Evaluation measure {distribution_evaluator} not supported for run_C45_combinations")
    with open(evaluationFunctionDir, "r") as f:
        eval_functions = f.readlines()
        eval_functions = [line.replace("\n", "") for line in eval_functions]
    
    print(f"eval_func {eval_functions}")
   
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
    file_dataset = FileDataset(trainFile)
    dt_builder = DecisionTreeBuilder(file_dataset, X_train, y_train)
    dt_builder.distributionEvaluator = get_distribution_evaluator(distribution_evaluator)

    dt_builder.FeatureCount = int(math.log(len(file_dataset.Attributes), 2) + 1)
    dt_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition

    func_combinations = list(itertools.combinations(eval_functions, combination_size))
    for combination in func_combinations:
        dt_builder.distributionEvaluator = dt_builder.distributionEvaluator(list(combination))
        dt = dt_builder.Build()
        dt_classifier = DecisionTreeClassifier(dt)
        y_scores = []
        for instance in X_test:
            inst_classify = dt_classifier.Classify(instance)
            #print(f"x_classify: {inst_classify}" )
            y_scores.append(inst_classify)
    
        y_pred = [ArgMax(instance) for instance in y_scores]
        #print(f"y_pred: {y_pred}")
        confusion, acc, auc = score_txtfile(y_pred, y_test, file_dataset)

        WriteResultsCSV(confusion, acc, auc, 0, testFile, outputDirectory, resultsId, "Not applicable", distribution_evaluator,
        functions_to_combine=list(combination))
        show_results(confusion, acc, auc, 0)
        dt_builder.distributionEvaluator = get_distribution_evaluator(distribution_evaluator)



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

def score_txtfile(predicted, y, dataset):
    real = list(map(lambda instance: dataset.GetClassValue(instance), y))
    numClasses = len(dataset.Class[1])
    confusion = [[0]*numClasses for i in range(numClasses)]
    classified_as = 0
    error_count = 0

    for i in range(len(real)):
        if real[i] != predicted[i]:
            error_count = error_count + 1
        confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

    acc = 100.0 * (len(real) - error_count) / len(real)
    auc = obtainAUCMulticlass(confusion, numClasses)

    return confusion, acc, auc

def Train_and_test(X_train, y_train, X_test, y_test, treeCount, multivariate, filtering, dataset=None):
    classifier = PBC4cip(tree_count=treeCount, filtering = filtering, file_dataset=dataset)
    patterns = classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    confusion, acc, auc = classifier.score(y_pred, y_test)

    
    return patterns, confusion, acc, auc

def test_PBC4cip(trainFile, outputDirectory, treeCount, multivariate, filtering, testFile, resultsId, delete, distribution_evaluator): 
    #Uncomment this to work with text files instead of dataframes  
    
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
    file_dataset = FileDataset(trainFile)
    classifier = PBC4cip(tree_count=treeCount, multivariate=multivariate, filtering=filtering, file_dataset=trainFile, distribution_evaluator=distribution_evaluator)
    patterns = classifier.fit(X_train, y_train)
    y_test_scores = classifier.score_samples(X_test)
    #print(f"y test scores: {y_test_scores}")
    y_pred = classifier.predict(X_test)
    #print(f"y_pred: {y_pred}")
    confusion, acc, auc = score_txtfile(y_pred, y_test, FileDataset(trainFile))
    for pattern in patterns:
        #print(f"patt: {pattern}")
        pass
    """
    train_df, test_df = import_data(trainFile, testFile)
    X_train, y_train, X_test, y_test = split_data(train_df, test_df)
    classifier = PBC4cip(tree_count=treeCount, multivariate=multivariate, filtering=filtering, distribution_evaluator=distribution_evaluator)
    patterns = classifier.fit(X_train, y_train)
    for pattern in patterns:
        #print(f"patt: {pattern}")
        pass

    y_test_scores = classifier.score_samples(X_test)
    
    y_pred = classifier.predict(X_test)
    confusion, acc, auc = score(y_pred, y_test)
    """
    #WritePatternsCSV(patterns, trainFile, outputDirectory)
    #WritePatternsBinary(patterns, trainFile, outputDirectory)
    WriteResultsCSV(confusion, acc, auc, len(patterns), testFile, outputDirectory, resultsId, filtering, distribution_evaluator)
    show_results(confusion, acc, auc, len(patterns))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        #test_PBC4cip(training_files[f], args.output_directory, args.tree_count, args.multivariate,
            #args.filtering,  testing_files[f], resultsId, args.delete_binary, args.distribution_evaluation )
        #run_C45(training_files[f], args.output_directory,  testing_files[f], resultsId, args.distribution_evaluation
        #, args.evaluation_functions)
        #run_C45_combinations(training_files[f], args.output_directory,  testing_files[f], resultsId, args.distribution_evaluation
        #, args.evaluation_functions, args.combination_size)
        #run_C45_multiple(training_files[f], args.output_directory,  testing_files[f], resultsId
        #, args.evaluation_functions, args.evaluation_functions_list)
        wilcoxon(training_files[f], args.output_directory)
        #order_results(training_files[f], args.column_names, args.output_directory)
        

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

    parser.add_argument("--multivariate",
                        type=str2bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help="states if multivariate tree builder variant is to be used")

    parser.add_argument("--delete-binary",
                        type=str2bool,
                        default=True,
                        nargs='?',
                        help="states if binary file is to be deleted after execution")

    parser.add_argument("--tree-count",
                        type=int,
                        metavar="n",
                        default=100,
                        help="indicates the number of trees that will be used")

    parser.add_argument("--filtering",
                        type = str2bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help="Decides wether the found patterns are to be filtered or not")

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
    
    parser.add_argument("--distribution-evaluation",
                        type=str,
                        metavar="distEval",
                        default='quinlan',
                        help="indicates the evaluation function used for the tree")
    
    parser.add_argument("--evaluation-functions",
                        type=str,
                        metavar="evalFuncs",
                        default=None,
                        help="indicates which functions to be combined if so needed")
    parser.add_argument("--evaluation-functions-list",
                        type=str,
                        metavar="evalFuncsList",
                        default=None,
                        help="indicates which functions to be used to run C45 multiple times")                    
    parser.add_argument("--combination-size",
                        type=int,
                        metavar='k',
                        default=2,
                        help="indicates the size of the combination for the evaluation functions")
    parser.add_argument("--column-names",
                        type=str,
                        metavar='colNames',
                        default=None,
                        help="Gets the column names from a csv file")
    

    

    args = parser.parse_args()

    print("==========================================================")
    print("         o--o  o--o    o-o o  o                           ")
    print("         |   | |   |  /    |  |      o                    ")
    print("         O--o  O--o  O     o--O  o-o   o-o                ")
    print("         |     |   |  \       | |    | |  |               ")
    print("         o     o--o    o-o    o  o-o | O-o                ")
    print("                                       |                  ")
    print("                                       o                  ")
    print("==========================================================")

    if not args.training_files and not args.training_directory and not args.input_files and not args.input_directory:
        parser.print_help()
    else:
        Execute(args)
