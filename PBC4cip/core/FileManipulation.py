import os
import pickle
import csv
from tqdm import tqdm
from io import StringIO, BytesIO
import pandas as pd
import numpy as np

def ReadARFF(file):
    import arff
    return arff.loads(open(file))


def ReadAndDeleteARFF(file):
    dataset = ReadARFF(file)
    if os.path.exists(file):
        os.remove(file)
    else:
        print(f"Can not delete the file '{file}' as it doesn't exists")
    return dataset

def setClassAttribute(file, output_name):
    f = open(file).readlines()
    if os.path.exists(file):
        os.remove(file)
    else:
        raise Exception(f"file {file} did have its class attribute properly set")
    new_f = str( file[:len(file)-9]+'.arff')
    with open (new_f, 'w') as arff_file:
        for line in f:
            if line != '\n':
                newline = line
                if newline.split()[0] == '@attribute' and newline.split()[1] == output_name:
                    newline = newline.replace(output_name, 'Class')
                arff_file.write(newline)
    return ReadAndDeleteARFF(new_f)

def ReadDAT(file):
    f = open(file).readlines()
    new_f = str(file+'-copy.dat')
    with open(new_f, 'w') as new_file:
        for line in f:
            if line != '\n':
                newline = line
                if newline.split()[0] == '@inputs':
                    continue
                if newline.split()[0] == '@output' or newline.split()[0] == '@outputs':
                    output_name = newline.split()[1]
                elif newline.split()[0] == '@attribute':
                    newline = newline.replace(
                        'real', 'real ').replace(
                        'REAL', 'REAL ').replace('integer', 'integer ').replace(
                            'INTEGER', 'INTEGER ').replace('  ', ' ')
                    if '{' in newline.split()[1]:
                        newline = newline.replace('{', ' {')
                    if newline.split()[2].lower() in ['numeric', 'real', 'integer', 'string']:
                        newline = newline.split(
                        )[0]+' '+newline.split()[1]+' '+newline.split()[2]+'\n'
                new_file.write(newline)
    return setClassAttribute(new_f, output_name)

def convert_dat_to_csv(file):
    f = open(file).readlines()
    new_f = str(file[0:len(file)-4]+'.csv')
    column_names = []
    with open(new_f, 'w') as new_file:
        for line in f:
            if line != '\n':
                newline = line
                if newline.split()[0] == '@outputs' or newline.split()[0] == '@output' or newline.split()[0] == '@attribute' \
                or newline.split()[0] == '@relation' or newline.split()[0] == '@data' or newline.split()[0] == '@relation':
                    continue
                elif newline.split()[0] == '@inputs' or newline.split()[0] == '@input':
                    newline = newline.replace(
                        '@inputs', '@input'
                    ).replace('@input', '').replace(' ', '').replace('\n','')
                    newline = newline + ',Class\n'
                else :
                    newline.replace(' ', '')
                new_file.write(newline)

    return new_f


def ReadPatternsBinary(originalFile, outputDirectory, delete, suffix=None):
    patterns = list()

    if not suffix:
        suffix = ""

    name = os.path.splitext(os.path.basename(originalFile))[0]
    name = os.path.join(
        outputDirectory, name[:len(name)-len(suffix)]+'.pypatterns')

    if os.path.exists(name):
        input_file = open(name, "rb")
        patternCount = pickle.load(input_file)
        for pattern in tqdm(range(patternCount), desc=f"Reading patterns from {name}", unit="pat", leave=False):
            try:
                pattern_in = pickle.load(input_file)
                patterns.append(pattern_in)
            except EOFError:
                break
    else:
        raise Exception(
            f"File '{name}'' not found! Please extract patterns first!")
    return patterns


def WritePatternsBinary(patterns, originalFile, outputDirectory, suffix=None):
    if not patterns or len(patterns) == 0:
        return ""
    if not suffix:
        suffix = ""
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    name = os.path.splitext(os.path.basename(originalFile))[0]
    name = os.path.join(
        outputDirectory, name[:len(name)-len(suffix)]+'.pypatterns')

    action = "Writing"
    if os.path.exists(name):
        action = "Overwriting"
        os.remove(name)

    patterns_out = open(name, "wb")
    pickle.dump(len(patterns), patterns_out)
    for pattern in tqdm(patterns, desc=f"{action} patterns to {name}...", unit="pattern", leave=False):
        pickle.dump(pattern, patterns_out)
        patterns_out.flush()
    patterns_out.close()

    return name


def WritePatternsCSV(patterns, originalFile, outputDirectory, suffix=None):
    if not patterns or len(patterns) == 0:
        return ""
    if not suffix:
        suffix = ""
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    name = os.path.splitext(os.path.basename(originalFile))[0]
    name = os.path.join(outputDirectory, name[:len(name)-len(suffix)]+'.csv')

    action = "Writing"
    if os.path.exists(name):
        action = "Overwriting"
        os.remove(name)

    patterns_out = open(name, "w", newline='\n', encoding='utf-8')
    fields = list(patterns[0].ToString().keys())
    pattern_writer = csv.DictWriter(patterns_out, fieldnames=fields)
    pattern_writer.writeheader()
    for pattern in tqdm(patterns, desc=f"{action} patterns to {name}...", unit="pattern", leave=False):
        pattern_writer.writerow(pattern.ToString())

    patterns_out.close()

    return name

def WriteResultsCSV(confusion, acc, auc, numPatterns, originalFile, outputDirectory, resultsId, filtering, distribution_evaluator,
functions_to_combine=None ):
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    datasetName = os.path.splitext(os.path.basename(originalFile))[0]
    name = os.path.join(outputDirectory, f"TestsResults{resultsId}.csv")

    action = "Writing"
    if os.path.exists(name):
        action = "Appending"
        results_out = open(name, "a+", newline='\n', encoding='utf-8')
    elif functions_to_combine is None:
        results_out = open(name, "w+", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC,Acc,NumPatterns,Filtering,distribution_evaluator\n")
    else:
        results_out = open(name, "w+", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC,Acc,NumPatterns,Filtering,distribution_evaluator,eval_functions\n")


    if functions_to_combine is None:
        results_out.write(f"{datasetName},{str(auc)}, {str(acc)}, {str(numPatterns)}, {str(filtering)}, {str(distribution_evaluator)}\n")
    else:
        results_out.write(f"{datasetName},{str(auc)}, {str(acc)}, {str(numPatterns)}, {str(filtering)}, {str(distribution_evaluator)}, {'-'.join(functions_to_combine)}\n")
    results_out.close()

    return name

def GetFromFile(file):
    if os.path.isfile(file):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".arff":
            return ReadARFF(file)
        elif file_extension == ".dat":
            return ReadDAT(file)
        else:
            raise Exception(
                f"Extension '{file_extension}' of file '{filename}' is not supported ")
    else:
        raise Exception(f"File: {file} is not valid")

def returnX_y(file):
        arff_file = GetFromFile(file)
        instancesDf = get_dataframe_from_arff(arff_file)
        instances = instancesDf.to_numpy()
        X = instances[:, 0:len(instances[0])-1]
        y = instances[:, len(instances[0])-1 : len(instances[0])]
        return X,y

def get_dataframe_from_arff(arff_file):
    instancesDf = pd.DataFrame.from_records(
            arff_file['data'], columns=list(
                map(lambda attribute: attribute[0], arff_file['attributes'])
            )
        )
    instancesDf = instancesDf.fillna(value=np.nan)
    return instancesDf
