import os
import argparse

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary, WriteClassificationResults, WriteResultsCSV, returnX_y
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from PatternMiner import PatternMinerWithoutFiltering
from core.PatternFilter import MaximalPatternsGlobalFilter
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate
from core.Helpers import ArgMax
from core.Dataset import Dataset
from datetime import datetime


def show_results(confusion, acc, auc, numPatterns):
    print()
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"acc: {acc} , auc: {auc} , numPatterns: {numPatterns}")

def prediction(X, y, patterns, classifier, dataset):
        if not patterns or len(patterns) == 0:
            raise Exception(
                "In order to classify, previously extracted patterns are required.")

        classification_results = list()

        for instance in tqdm(X, desc=f"Classifying instances", unit="instance", leave=False):
            result = classifier.predict(instance)
            classification_results.append(result)

        
        real = list(map(lambda instance: dataset.GetClassValue(instance), y))
        print("\n\nClassification_Results:")
        for result in classification_results:
            print(f"{result}")
        predicted = [ArgMax(instance) for instance in classification_results]

        return Evaluate(dataset.Class[1], real, predicted)

def Train_and_test(X_train, y_train, X_test, y_test, treeCount, multivariate, filtering, dataset):
    classifier = PBC4cip(treeCount)
    miner = PatternMinerWithoutFiltering()
    miner.dataset = dataset
    classifier.miner = miner
    if filtering:
        filterer = MaximalPatternsGlobalFilter()
        classifier.filterer = filterer

    classifier.multivariate = multivariate
    if multivariate:
        miner.decisionTreeBuilder = MultivariateDecisionTreeBuilder(dataset, X_train, y_train)
        miner.decisionTreeBuilder.distributionEvaluator = QuinlanGain
    else:
        miner.decisionTreeBuilder = DecisionTreeBuilder(dataset, X_train, y_train)
        miner.decisionTreeBuilder.distributionEvaluator = Hellinger
    classifier.dataset = dataset
    patterns = classifier.fit(X_train, y_train)
    confusion, acc, auc = prediction(X_test, y_test, patterns, classifier, dataset)
    return patterns, confusion, acc, auc

def runPBC4cip(trainFile, outputDirectory, treeCount, multivariate, filtering, testFile, resultsId, delete):
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
    """
    print(f"X_train: {X_train}")
    print(f"y_train:")
    for i,_ in enumerate(y_train):
        print(f"{y_train[i]}", end ="," )
    print(f"X_test: {X_test}")
    print(f"y_test:")
    for i,_ in enumerate(y_test):
        print(f" {y_test[i]}", end ="," )
    """
    dataset = Dataset(trainFile)
    patterns, confusion, acc, auc = Train_and_test(X_train, y_train, X_test, y_test, treeCount, multivariate, filtering, dataset)
    WritePatternsCSV(patterns, trainFile, outputDirectory)
    WritePatternsBinary(patterns, trainFile, outputDirectory)
    WriteResultsCSV(confusion, acc, auc, len(patterns), testFile, outputDirectory, resultsId, filtering)
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
