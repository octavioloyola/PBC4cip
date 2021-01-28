import os
import argparse
import numpy as np

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary, WriteClassificationResults
from core.FileManipulation import WriteResultsCSV, returnX_y, get_dataframe_from_arff, GetFromFile
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMinerWithoutFiltering
from core.PatternFilter import MaximalPatternsGlobalFilter
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate
from core.Helpers import ArgMax, convert_to_ndarray
from core.Dataset import Dataset, FileDataset, PandasDataset
from datetime import datetime


def show_results(confusion, acc, auc, numPatterns):
    print()
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"acc: {acc} , auc: {auc} , numPatterns: {numPatterns}")

def Train_and_test(X_train, y_train, X_test, y_test, treeCount, multivariate, filtering, dataset=None):
    classifier = PBC4cip(tree_count=treeCount, filtering = filtering, file_dataset=dataset)
    patterns = classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    confusion, acc, auc = classifier.score(y_pred, y_test)
    return patterns, confusion, acc, auc

def runPBC4cip(trainFile, outputDirectory, treeCount, multivariate, filtering, testFile, resultsId, delete):
    #Uncomment for testing obtaining model information and datasets from a file
    
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
    dataset = FileDataset(trainFile)
    
    
    arff_file_train = GetFromFile(trainFile)
    X_train_df = get_dataframe_from_arff(arff_file_train)
    X_train_df.rename(columns = {'class':'Class'}, inplace = True) 
    y_train_df = X_train_df.pop('Class')

    arff_file_test = GetFromFile(testFile)
    X_test_df = get_dataframe_from_arff(arff_file_test)
    X_test_df.rename(columns = {'class':'Class'}, inplace = True) 
    y_test_df = X_test_df.pop('Class')
    


    patterns, confusion, acc, auc = Train_and_test(X_train_df, y_train_df, X_test_df, y_test_df, treeCount, multivariate, filtering)
    #WritePatternsCSV(patterns, trainFile, outputDirectory)
    #WritePatternsBinary(patterns, trainFile, outputDirectory)
    #WriteResultsCSV(confusion, acc, auc, len(patterns), testFile, outputDirectory, resultsId, filtering)
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
    now = datetime.now()
    resultsId = now.strftime("%Y%m%d%H%M%S")
    runPBC4cip(args.training_files[0], args.output_directory, args.tree_count, args.multivariate,
        args.filtering, args.input_files[0], resultsId, args.delete_binary)


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

    parser.add_argument("--input-files",
                        type=str,
                        metavar="<*.dat/*.arff>",
                        nargs="+",
                        help="a file or files to be classified")

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
    

    args = parser.parse_args()
    if not args.training_files and not args.training_directory and not args.input_files and not args.input_directory:
        parser.print_help()
    else:
        Execute(args)
