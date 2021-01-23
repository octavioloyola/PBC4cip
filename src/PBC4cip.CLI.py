import os
import argparse
from tqdm import tqdm, trange
from core.PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary, WriteClassificationResults, WriteResultsCSV, returnX_y
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMinerWithoutFiltering
from core.PatternFilter import MaximalPatternsGlobalFilter
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate
from core.Helpers import ArgMax
from core.Dataset import Dataset
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
        miner.decisionTreeBuilder = MultivariateDecisionTreeBuilder(dataset)
        miner.decisionTreeBuilder.distributionEvaluator = QuinlanGain
    else:
        miner.decisionTreeBuilder = DecisionTreeBuilder(dataset)
        miner.decisionTreeBuilder.distributionEvaluator = Hellinger
    classifier.dataset = dataset
    patterns = classifier.fit(X_train, y_train)
    confusion, acc, auc = prediction(X_test, y_test, patterns, classifier, dataset)
    return patterns, confusion, acc, auc

def runPBC4cip(trainFile, outputDirectory, treeCount, multivariate, filtering, testFile, resultsId, delete):
    X_train, y_train = returnX_y(trainFile)
    X_test, y_test = returnX_y(testFile)
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
        runPBC4cip(training_files[f], args.output_directory, args.tree_count, args.multivariate,
            args.filtering,  testing_files[f], resultsId, args.delete_binary )


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
