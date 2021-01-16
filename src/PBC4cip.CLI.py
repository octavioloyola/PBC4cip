#!/usr/bin/env python
# coding: utf-8
import os
import argparse
from tqdm import tqdm, trange
from core.PBC4cip import PBC4cip
from core.FileManipulation import WritePatternsBinary, WritePatternsCSV, ReadPatternsBinary, WriteClassificationResults, WriteResultsCSV
from core.PatternMiner import PatternMiner, PatternMinerWithoutFiltering
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
    print("aaaa")
    print(directory)
    files = []
    if os.path.isdir(directory):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    else:
        raise Exception(f"Directory {directory} is not valid.")


def Train(file, outputDirectory, treeCount, multivariate, filtering, suffix=None):
    print(f"Training is about to begin")
    classifier = PBC4cip(file)
    patterns = classifier.Training(multivariate, filtering, treeCount)
    print(f"Patterns in Train are: {len(patterns)}")
    #WritePatternsCSV(patterns, file, outputDirectory, suffix) #Create the csv file with patterns
    WritePatternsBinary(patterns, file, outputDirectory, suffix) #Create the .pypatterns file


def Classify(file, outputDirectory, resultsId, delete, suffix=None):
    print("Testing is about to begin")

    try:
        classifier = PBC4cip()
        patterns = ReadPatternsBinary(file, outputDirectory, delete, suffix)
        confusion, acc, auc = classifier.Classification(patterns, testInstances)
        #WriteClassificationResults(confusion, acc, auc, file, outputDirectory, suffix)
        WriteResultsCSV(confusion, acc, auc, len(patterns), file, outputDirectory, resultsId, None)

        for i in range(len(confusion[0])):
            for j in range(len(confusion[0])):
                print(f"{confusion[i][j]} ", end='')
            print("")
        print(f"acc: {acc} , auc: {auc} , numPatterns: {len(patterns)}")
    except Exception as e:
        print(e)

def Train_and_test(trainFile, outputDirectory, treeCount, multivariate, filtering, testFile, resultsId, delete):
    print(f"Train and Test")
    classifier = PBC4cip()
    patterns = classifier.Training(multivariate, filtering, trainFile, treeCount )
    print(f"Now start testing")
    confusion, acc, auc = classifier.Classification(patterns, testFile)
    print(f"TrainLen: {trainFile} testLen: {testFile}")
    WriteResultsCSV(confusion, acc, auc, len(patterns), testFile, outputDirectory, resultsId, filtering)

    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"acc: {acc} , auc: {auc} , numPatterns: {len(patterns)}")


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
    # Initialize the array of testing and training dataset files

    # region File lists initialization

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

   # endregion

    print("===============================================================================")
    if len(training_files) == len(testing_files):
        tra = trange(len(training_files), desc='Training and Testing Files...', leave=True, unit="dataset")

        now = datetime.now()  # current date and time
        resultsId = now.strftime("%Y%m%d%H%M%S")

        for f in tra:
            tra.set_description(f"Working from {training_files[f]}")
            Train_and_test(training_files[f], args.output_directory, args.tree_count, args.multivariate,
                args.filtering,  testing_files[f], resultsId, args.delete_binary )
    else:
        tra = trange(len(training_files), desc='Training files...',
                    leave=True, unit="dataset")
        # tqdm(training_files, desc="Training files...", unit="dataset"):
        for f in tra:
            tra.set_description(f"Extracting patterns from {training_files[f]}")
            tra.refresh()  # to show immediately the update
            Train(training_files[f], args.output_directory, args.tree_count, args.multivariate,
                args.filtering, args.training_file_suffix)

        print("===============================================================================")
        tst = trange(len(testing_files), desc='Testing files...',
                    leave=True, unit="dataset")
        # tqdm(testing_files, desc="Testing patterns...", unit="dataset"):

        now = datetime.now()  # current date and time
        resultsId = now.strftime("%Y%m%d%H%M%S")

        for f in tst:
            tst.set_description(f"Classifying instances from {testing_files[f]}")
            tst.refresh()  # to show immediately the update
            Classify(testing_files[f], args.output_directory, resultsId,
                    args.delete_binary, args.test_file_suffix)
            #tst.set_description(f"Results saved for {testing_files[f]}")
            tst.refresh()  # to show immediately the update


if __name__ == '__main__':

    defaultDataDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "data", "python")
    defaultOutputDir = os.path.join(os.path.normpath(
        os.path.join(os.getcwd(), os.pardir)), "output")

    defaultTrainingFiles = list()
    defaultTestingFiles = list()

    # Some samples to test the program (they are already in the data folder)
    # region Default files
    # defaultTrainingFiles.append(os.path.join(os.path.normpath(
    #     defaultDataDir), "winequality-white-3_vs_7tra.dat"))
    # defaultTestingFiles.append(os.path.join(os.path.normpath(
    #     defaultDataDir), "winequality-white-3_vs_7tst.dat"))
    # endregion
    parser = argparse.ArgumentParser(
        description="Process some class imbalace datasets using PBC4cip.")

    parser.add_argument("--training-files",
                        type=str,
                        metavar="<*.dat/*.arff>",
                        # default=defaultTrainingFiles,
                        nargs="+",
                        help="a file or files that are going to be used to train the classifier")

    parser.add_argument("--training-directory",
                        type=str,
                        metavar="'"+defaultDataDir+"'",
                        # default=defaultDataDir,
                        help="the directory with files to be used to train the classifier")

    parser.add_argument("--input-files",
                        type=str,
                        metavar="<*.dat/*.arff>",
                        # default=defaultTestingFiles,
                        nargs="+",
                        help="a file or files to be classified")

    parser.add_argument("--input-directory",
                        type=str,
                        metavar="'"+defaultDataDir+"'",
                        #default=defaultDataDir,
                        help="the directory with files to be classified")

    parser.add_argument("--output-directory",
                        type=str,
                        metavar="'"+defaultOutputDir+"'",
                        default=defaultOutputDir,
                        # default="\\root\\parentDir\\dir",
                        help="the output directory for the patterns")

    parser.add_argument("--multivariate",
                        type=str2bool,
                        const=True,
                        default=False,
                        nargs='?',
                        help="states if multivariate variant is to be used")

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
    print()
    # Arguments received: Uncomment for debbuging purposes
    # print()
    # Uncomment for debbuging purposes
    print()

    print(f"Args: \n {args} \n\n")
    if not args.training_files and not args.training_directory and not args.input_files and not args.input_directory:
        parser.print_help()
    else:
        Execute(args)
