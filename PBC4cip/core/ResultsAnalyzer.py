import csv
import math
import os
import random
import statistics

import baycomp
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from .Evaluation import obtainAUCMulticlass


def show_results(confusion, acc, auc, numPatterns):
    print()
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            pass
            #print(f"{confusion[i][j]} ", end='')
        #print("")
    print(f"acc: {acc} , auc: {auc} , numPatterns: {numPatterns}")

def join_prelim_results(fileDir, outputDirectory):
    df = pd.read_csv(fileDir)

    outputDirectory = outputDirectory + "\\joined-results"

    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    print(f"FileName: {os.path.splitext(os.path.basename(fileDir))}")

    FileName = os.path.splitext(os.path.basename(fileDir))[0].split('-')[0]
    name = os.path.join(outputDirectory, f"-{FileName}.csv")

    action = "Writing"
    if os.path.exists(name):
        action = "Appending"
        results_out = open(name, "a+", newline='\n', encoding='utf-8')
    elif 'eval_functions' in df.columns:
        results_out = open(name, "w+", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC,Acc,NumPatterns,Filtering,distribution_evaluator,eval_functions\n")
    else:
        results_out = open(name, "w+", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC,Acc,NumPatterns,Filtering,distribution_evaluator\n")
    
    for i,row in df.iterrows():
        if 'eval_functions' in df.columns:
            results_out.write(f"{str(row['File'])},{str(row['AUC'])},{str(row['Acc'])},{str(row['NumPatterns'])},{str(row['Filtering'])},{str(row['distribution_evaluator'])},{str(row['eval_functions'])}\n")
        else:
            results_out.write(f"{str(row['File'])},{str(row['AUC'])},{str(row['Acc'])},{str(row['NumPatterns'])},{str(row['Filtering'])},{str(row['distribution_evaluator'])}\n")
    results_out.close()

    return name


def order_results(fileDir, output_directory):
    df = pd.read_csv(fileDir)
    output_directory = output_directory + "\\order-results"

    if not os.path.exists(output_directory):
        print(f"Creating order-results directory: {output_directory}")
        os.makedirs(output_directory)
    
    file_name = os.path.splitext(os.path.basename(fileDir))[0]
    file_name = os.path.join(output_directory, file_name+'-ordered.csv')

    action = "Writing"
    if os.path.exists(file_name):
        action = "Overwriting"
        os.remove(file_name)
    
    if (str(df.at[0, 'distribution_evaluator'].strip()) in ['combiner', 'combiner-random', 'irv', 'schulze', 
                                                            'coombs', 'bucklin', 'reciprocal', 'stv']):
        column_names = df.eval_functions.unique()
        file_names = df.File.unique()
        print(f"len file: {len(file_names)}")
        df_output = pd.DataFrame()
        df_output['File'] = file_names
        for name in column_names:
            temp_auc = list(df[df['eval_functions'] == name]['AUC'])
            df_output[f'{name}-AUC'] = temp_auc
            temp_acc =list(df[df['eval_functions'] == name]['Acc'])
            df_output[f'{name}-Acc'] = temp_acc
    else:
        print(str(df.at[0, 'distribution_evaluator'].strip()))
        column_names = df.distribution_evaluator.unique()
        file_names = df.File.unique()
        print(f"len file: {len(file_names)}")
        df_output = pd.DataFrame()
        df_output['File'] = file_names
        for name in column_names:
            temp_auc = list(df[df['distribution_evaluator'] == name]['AUC'])
            df_output[f'{name}-AUC'] = temp_auc
            temp_acc =list(df[df['distribution_evaluator'] == name]['Acc'])
            df_output[f'{name}-Acc'] = temp_acc

    csv_data = df_output.to_csv(file_name, index = False)

    return file_name

def transpose_results(fileDir, column_names, output_directory):
    df = pd.read_csv(fileDir)
    with open(column_names, "r") as f:
        col_names = f.readlines()
        col_names = [line.replace("\n", "").strip() for line in col_names]
    
    print(f"col_names: {len(col_names)}")
    print(f"{col_names}")
    file_name = os.path.splitext(os.path.basename(fileDir))[0]
    file_name = os.path.join(output_directory, file_name+'-transpose.csv')

    action = "Writing"
    if os.path.exists(file_name):
        action = "Overwriting"
        os.remove(file_name)

    sep_cols = []
    for name in col_names:
        sep_cols.append(name+'-AUC')
        sep_cols.append(name+'-Acc')

    print(f"sep_cols: {len(sep_cols)}")
    results_out = open(file_name, "a", newline='\n', encoding='utf-8')
    results_out.write('File,'+",".join(sep_cols)+"\n")
    

    print(f"lenn: {len(col_names)}")
    val = len(df.index) // len(col_names)
    print(f"val:{len(df.index) // len(col_names)}")

    for i in range(val):
        lst = [i+(idx*val) for idx in range(len(col_names))]
        auc_result = [str(df.at[i+(idx*val), 'AUC']) for idx in range(len(col_names))]
        acc_result = [str(df.at[i+(idx*val), 'Acc']) for idx in range(len(col_names))]
        result = [None]*(len(auc_result) + len(acc_result))
        result[::2] = auc_result
        result[1::2] = acc_result
        results_out.write(str(df.at[i,'File'])+','+",".join(result)+"\n")
    results_out.close()

    return file_name

def wilcoxon(fileDir, output_directory):
    df = pd.read_csv(fileDir)
    num_df = df.drop(columns=['File'])
    
    output_directory = output_directory + "\\stat-tests"

    if not os.path.exists(output_directory):
        print(f"Creating stat test directory: {output_directory}")
        os.makedirs(output_directory)
        
    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(output_directory, auc_name +'-wilcoxon.csv')

    action = "Writing"
    if os.path.exists(auc_name):
        action = "Overwriting"
        os.remove(auc_name)
    
    auc_results_out = open(auc_name, "a", newline='\n', encoding='utf-8')
    auc_results_out.write(f"Combination,P-Value,W-Pos,W-Neg\n")

    for col_x in num_df.columns:
        for col_y in num_df.columns:
            if col_x != col_y:
                combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
                x,y = map(np.asarray, (num_df[f'{col_x}'],num_df[f'{col_y}']))
                d = x-y
                d = np.compress(np.not_equal(d, 0), d)
                if len(d) == 0:
                    auc_results_out.write(f"{combination},{str(1)},{str(0)},{str(0)}\n")
                    continue

                r = stats.rankdata(abs(d))
                r_plus = np.sum((d > 0)* r)
                r_minus = np.sum((d < 0) * r)

                w,p = stats.wilcoxon(x,y)
                auc_results_out.write(f"{combination},{str(p)},{str(r_plus)},{str(r_minus)}\n")
        auc_results_out.write(f"\n")

    return auc_name

def analyze_wilcoxon(fileDir, outputDirectory):
    print(f"Analyze Wilcoxon:\n\n: {fileDir}")
    df_wilcoxon = pd.read_csv(fileDir)
    print(f"df_wilcoxon: {df_wilcoxon.head(10)}")

    df_wilcoxon = df_wilcoxon.dropna()

    outputDirectory = outputDirectory + "//stat-tests"
    name = os.path.splitext(os.path.basename(fileDir))[0]
    name = os.path.join(outputDirectory, name +'-final.csv')

    results_out = open(name, "w+", newline='\n', encoding='utf-8')
    results_out.write('Combination,Better-Than\n')
    

    row_df = next(df_wilcoxon.iterrows())[1]
    curr_comb_name = str(row_df['Combination']).split('vs')[0].strip()
    comb_better_amt = 0

    for i,row in df_wilcoxon.iterrows():
        
        comb_name_right = str(row['Combination']).split('vs')[1].strip()
        comb_name_left = str(row['Combination']).split('vs')[0].strip()
        if comb_name_left != curr_comb_name:
            results_out.write(f'{str(curr_comb_name)},{str(comb_better_amt)}\n')
            comb_better_amt = 0
            curr_comb_name = comb_name_left
        if "-" not in comb_name_right:
            if float(row['P-Value']) < 0.05 and float(row['W-Pos']) > float(row['W-Neg']):
                #print(f"comb: {comb_name_left} vs: {comb_name_right} p: {float(row['P-Value'])}")
                comb_better_amt = comb_better_amt + 1
        
    results_out.write(f'{str(curr_comb_name)},{str(comb_better_amt)}\n')
    results_out.close()

    return name

def one_bayesian_one(fileDir, k, output_directory, runs=1):
    df = pd.read_csv(fileDir)
    num_df = df.drop(columns=['File'])
    iterations = df.shape[0] // (k * runs)

    p_left_lst = []
    p_rope_lst = []
    p_right_lst = []
    for i in range(iterations):
        x,y = map(np.array, (num_df.iloc[i*k*runs:((i+1)*k*runs),0], num_df.iloc[i*k*runs:((i+1)*k*runs),1]))
        left, rope, right = baycomp.two_on_single(x,y, rope=0.01, runs=runs)
        p_left_lst.append(left)
        p_rope_lst.append(rope)
        p_right_lst.append(right)
        print(f"{left},{rope},{right}")
    
    output_directory = output_directory + "\\bayesian-tests"

    if not os.path.exists(output_directory):
        print(f"Creating bayesian test directory: {output_directory}")
        os.makedirs(output_directory)
        
    result_name = os.path.splitext(os.path.basename(fileDir))[0]
    print(f"len: {len(os.path.splitext(os.path.basename(fileDir)))}")
    print(f"{os.path.splitext(os.path.basename(fileDir))}")
    result_name = os.path.join(output_directory, result_name +'-bayes-single.csv')

    if os.path.exists(result_name):
        os.remove(result_name)

    results_out = open(result_name, "a", newline='\n', encoding='utf-8')
    results_out.write(f"File,P-Left,P-ROPE,P-Right\n")
    for i in range(iterations):
        results_out.write(f"{str(df.iloc[i*k*runs,0])},{str(p_left_lst[i])},{str(p_rope_lst[i])},{str(p_right_lst[i])}\n")

def one_bayesian_multiple(fileDir, k, output_directory, runs=1):
    df = pd.read_csv(fileDir)
    num_df = df.drop(columns=['File'])
    iterations = 1

    p_left_lst = []
    p_rope_lst = []
    p_right_lst = []
    for i in range(iterations):
        x,y = map(np.array, (num_df.iloc[i*k:((i+1)*k),0], num_df.iloc[i*k:((i+1)*k),1]))
        left, rope, right = baycomp.two_on_multiple(x,y, rope=0.01, runs=runs)
        p_left_lst.append(left)
        p_rope_lst.append(rope)
        p_right_lst.append(right)
        print(f"{left},{rope},{right}")
    
    output_directory = output_directory + "\\bayesian-tests"

    if not os.path.exists(output_directory):
        print(f"Creating bayesian test directory: {output_directory}")
        os.makedirs(output_directory)
        
    result_name = os.path.splitext(os.path.basename(fileDir))[0]
    print(f"len: {len(os.path.splitext(os.path.basename(fileDir)))}")
    print(f"{os.path.splitext(os.path.basename(fileDir))}")
    result_name = os.path.join(output_directory, result_name +'-bayes-multiple.csv')

    if os.path.exists(result_name):
        os.remove(result_name)

    results_out = open(result_name, "a", newline='\n', encoding='utf-8')
    results_out.write(f"File,P-Left,P-ROPE,P-Right\n")
    for i in range(iterations):
        results_out.write(f"{str(df.iloc[i*k,0])},{str(p_left_lst[i])},{str(p_rope_lst[i])},{str(p_right_lst[i])}\n")


def multiple_bayesian_multiple(fileDir, output_directory, runs=1):
    print(f"runs:{runs}")
    df = pd.read_csv(fileDir)
    num_df = df.drop(columns=['File'])

    output_directory = output_directory + "\\bayesian-tests"

    if not os.path.exists(output_directory):
        print(f"Creating bayesian test directory: {output_directory}")
        os.makedirs(output_directory)
        
    result_name = os.path.splitext(os.path.basename(fileDir))[0]
    result_name = os.path.join(output_directory, result_name +'-bayes.csv')

    print(len(num_df.columns))

    action = "Writing"
    if os.path.exists(result_name):
        action = "Overwriting"
        os.remove(result_name)
    
    results_out = open(result_name, "a", newline='\n', encoding='utf-8')
    results_out.write(f"Combination,P-Left,P-ROPE,P-Right\n")

    for col_x in tqdm(num_df.columns, desc=f"Performing bayesian analysis...", unit="col_x", leave=False):
        for col_y in tqdm(num_df.columns, desc=f"vs {col_x}...", unit="col_y", leave=False):
            combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
            if col_x != col_y and ("-" not in col_y[0:len(col_y)-4]):
                x,y = map(np.asarray, (num_df[f'{col_x}'],num_df[f'{col_y}']))
                left,rope,right = baycomp.two_on_multiple(x,y, rope = 0.01, runs=runs)
                #left,rope,right = ((random.randint(1,10),random.randint(1,10),random.randint(1,10)))
                print(f"left: {left} rope: {rope} right: {right}")

                results_out.write(f"{combination},{str(left)},{str(rope)},{str(right)}\n")
        results_out.write(f"\n")
    results_out.close()

    return result_name

def leo_bayesian(fileDir, output_directory, runs=1):
    aucs = pd.read_csv(fileDir)
    classifiers = aucs.columns[1:]
    m = [[None for j in range(i+1, len(classifiers))] for i in range(len(classifiers)-1)]
    result_name = os.path.splitext(os.path.basename(fileDir))[0]
    result_name = os.path.join(output_directory, result_name +'-leo-bayes.csv')
    print(f"resultName: {result_name}")
    
    for i in tqdm(range(len(classifiers)-1)):
        for j in range(1, len(classifiers)-i):
            m[i][j-1] = baycomp.two_on_multiple(aucs[classifiers[i]], aucs[classifiers[i+j]], 0.01)
    c1 = []
    c2 = []
    wins = []
    losses = []
    for i in range(len(m)):
        for j in range(len(m) - i):
            c1.append(classifiers[i])
            c2.append(classifiers[j+i+1])
            wins.append(m[i][j][0])
            losses.append(m[i][j][2])

            c2.append(classifiers[i])
            c1.append(classifiers[j+i+1])
            wins.append(m[i][j][2])
            losses.append(m[i][j][0])
    df = pd.DataFrame({'c1': c1, 'c2': c2, 'pwin': wins, 'plose' : losses})
    print(f"in leo bayes:\n{df.head(15)}")
    csv_data = df.to_csv(result_name, index = False)
    return result_name

def leo_bayesian_figure(fileDir, output_directory, runs=1):
    aucs = pd.read_csv(fileDir)
    classifiers = aucs.columns[1:]
    m = [[None for j in range(i+1, len(classifiers))] for i in range(len(classifiers)-1)]
    name = os.path.splitext(os.path.basename(fileDir))[0]
    result_name = os.path.join(output_directory, name +'-leo-bayes-fig.csv')
    img_name = os.path.join(output_directory, name + '-comb-kgv.png')
    print(f"resultName: {result_name}")
    
    for i in tqdm(range(len(classifiers)-1)):
        for j in range(1, len(classifiers)-i):
            m[i][j-1], fig = baycomp.two_on_multiple(aucs[classifiers[i]], aucs[classifiers[i+j]], 0.01, plot=True, names=('kgv', 'tw-qg-mch-cs-bhy'))
            fig.savefig(img_name)
    c1 = []
    c2 = []
    wins = []
    losses = []
    for i in range(len(m)):
        for j in range(len(m) - i):
            c1.append(classifiers[i])
            c2.append(classifiers[j+i+1])
            wins.append(m[i][j][0])
            losses.append(m[i][j][2])

            c2.append(classifiers[i])
            c1.append(classifiers[j+i+1])
            wins.append(m[i][j][2])
            losses.append(m[i][j][0])
    df = pd.DataFrame({'c1': c1, 'c2': c2, 'pwin': wins, 'plose' : losses})
    print(f"in leo bayes:\n{df.head(15)}")
    csv_data = df.to_csv(result_name, index = False)
    return result_name
    

def separate(fileDir, output_directory):
    df = pd.read_csv(fileDir)
    auc_df = df.filter(regex='-AUC|File')
    acc_df = df.filter(regex='-Acc|File')
    
    output_directory = output_directory + "\\separate-results"

    if not os.path.exists(output_directory):
        print(f"Creating separate results directory: {output_directory}")
        os.makedirs(output_directory)
    
    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(output_directory, auc_name +'-auc.csv')

    acc_name = os.path.splitext(os.path.basename(fileDir))[0]
    acc_name = os.path.join(output_directory, acc_name +'-acc.csv')
    
    csv_data = auc_df.to_csv(auc_name, index = False)
    csv_data = acc_df.to_csv(acc_name, index = False)

    return auc_name, acc_name

def average_k_runs_cross_validation(fileDir, k, output_directory):
    #k is the amount of data related to a given dataset, so for 10*10 folds, k is 100
    df = pd.read_csv(fileDir)
    chunk_size = len(df['File']) // k
    print(f"lenFile: { len(df['File'])} chunkSize: {chunk_size}")
    auc_df = df.filter(regex='-AUC')
    acc_df = df.filter(regex='-Acc')

    auc_lst = []
    acc_lst = []

    for col_x in auc_df.columns:
        avg = [np.average(x) for x in np.split(auc_df[f'{col_x}'], chunk_size)]
        auc_lst.append(avg)

    for col_x in acc_df.columns:
        avg = [np.average(x) for x in np.split(acc_df[f'{col_x}'], chunk_size)]
        acc_lst.append(avg)

    output_directory = output_directory + "\\order-results"

    if not os.path.exists(output_directory):
        print(f"Creating order results directory: {output_directory}")
        os.makedirs(output_directory)

    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(output_directory, auc_name +'-auc-avg-k' + str(k) + '.csv')

    action = "Writing"
    if os.path.exists(auc_name):
        action = "Overwriting"
        os.remove(auc_name)
    
    auc_results_out = open(auc_name, "a", newline='\n', encoding='utf-8')
    auc_results_out.write('File,'+",".join(auc_df.columns)+"\n")
    for i in range(chunk_size):
            result = [str(auc_lst[idx][i]) for idx in range(len(auc_lst))]
            auc_results_out.write(str(df.at[i * k,'File'])+','+",".join(result)+"\n")
    auc_results_out.close()

    acc_name = os.path.splitext(os.path.basename(fileDir))[0]
    acc_name = os.path.join(output_directory, acc_name +'-acc-avg-k' + str(k) + '.csv')

    action = "Writing"
    if os.path.exists(acc_name):
        action = "Overwriting"
        os.remove(acc_name)
    
    acc_results_out = open(acc_name, "a", newline='\n', encoding='utf-8')
    acc_results_out.write('File,'+",".join(acc_df.columns)+"\n")
    for i in range(chunk_size):
            result = [str(acc_lst[idx][i]) for idx in range(len(acc_lst))]
            acc_results_out.write(str(df.at[i * k,'File'])+','+",".join(result)+"\n")
    acc_results_out.close()

    return auc_name, acc_name

def append_results(fileDir, dir_to_append, outputDirectory):
    df_comb = pd.read_csv(fileDir)
    df_original = pd.read_csv(dir_to_append)

    df_comb = df_comb.sort_values('File').reset_index(drop=True)
    print(f"df_comb_size after reset: {len(df_comb)}")
    df_original = df_original.sort_values('File').reset_index(drop=True)

    print(f"comb: {df_comb.head(10)}")

    df_comb = df_comb.drop(['File'], axis=1)

    for col_name in df_comb.columns:
        df_original[f'{col_name}'] = df_comb[f'{col_name}']

    print(f"dfff: {df_original.head(10)}")

    outputDirectory = outputDirectory + "\\combined-results"

    if not os.path.exists(outputDirectory):
        print(f"Creating combined results directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(outputDirectory, auc_name +'comb.csv')

    action = "Writing"
    if os.path.exists(auc_name):
        action = "Overwriting"
        os.remove(auc_name)
    
    csv_data = df_original.to_csv(auc_name, index = False)
    #csv_data = df_comb.to_csv(auc_name, index=False)
    print(f"append_results: {auc_name}")
    return auc_name

def sort_results(fileDir, outputDirectory):
    df_comb = pd.read_csv(fileDir)
    print(f"comb: {df_comb.head(10)}")
    df_comb = df_comb.sort_values('File').reset_index(drop=True)
    print(f"df_comb_size after reset: {len(df_comb)}")

    print(f"comb: {df_comb.head(10)}")
    outputDirectory = outputDirectory + "\\combined-results"

    if not os.path.exists(outputDirectory):
        print(f"Creating combined results directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(outputDirectory, auc_name +'comb.csv')

    action = "Writing"
    if os.path.exists(auc_name):
        action = "Overwriting"
        os.remove(auc_name)
    
    csv_data = df_comb.to_csv(auc_name, index=False)
    print(f"append_results: {auc_name}")
    return auc_name

def analyze_bayes(fileDir, outputDirectory):
    df_bayes = pd.read_csv(fileDir)

    df_bayes = df_bayes.dropna()

    outputDirectory = outputDirectory + "//bayesian-tests"
    name = os.path.splitext(os.path.basename(fileDir))[0]
    name = os.path.join(outputDirectory, name +'-final.csv')

    results_out = open(name, "w+", newline='\n', encoding='utf-8')
    results_out.write('Combination,Better-Than\n')
    

    row_df = next(df_bayes.iterrows())[1]
    curr_comb_name = str(row_df['Combination']).split('vs')[0].strip()
    comb_better_amt = 0

    for i,row in df_bayes.iterrows():
        
        comb_name_right = str(row['Combination']).split('vs')[1].strip()
        comb_name_left = str(row['Combination']).split('vs')[0].strip()
        if comb_name_left != curr_comb_name:
            results_out.write(f'{str(curr_comb_name)},{str(comb_better_amt)}\n')
            comb_better_amt = 0
            curr_comb_name = comb_name_left
        if "-" not in comb_name_right:
            if float(row['P-Left']) >= 0.95:
                comb_better_amt = comb_better_amt + 1
        
    results_out.write(f'{str(curr_comb_name)},{str(comb_better_amt)}\n')
    results_out.close()


def read_shdz_results(fileDir, fileName, outputDirectory):
    auc = None
    acc = None
    names = fileDir.split("-")
    with open(fileDir, "r") as f:
            for line in f:
                if line != '\n':
                    newline = line
                    if newline.split()[0] == 'AUC':
                        auc = newline.split()[1]
                    elif newline.split()[0] == 'ACC':
                        acc = newline.split()[1]
            
    
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    name = os.path.splitext(os.path.basename(fileName))[0]
    name = os.path.join(outputDirectory, name+'-shdz.csv')

    action = "Writing"
    if os.path.exists(name):
        action = "Appending"
        results_out = open(name, "a+", newline='\n', encoding='utf-8')
    else:
        results_out = open(name, "w", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC,Acc\n")

    results_out.write(f"{'-'.join(names[2:])},{str(auc)},{str(acc)}\n")
    results_out.close()
    return name

def shorten_name(col_name):
    col_name = col_name.lower()
    col_name = col_name.replace("-auc", "").replace("twoing", "tw").replace("quinlan gain", "qg").replace(
    "gini impurity", "gi").replace("multi class hellinger", "mch").replace("chi squared", "cs").replace(
    "g statistic", "gs").replace("marsh", "msh").replace("normalized gain", "ng").replace("kolmogorov", "kgv").replace(
        "bhattacharyya", "bhy")

    return col_name

def convert_names(fileDir, types, outputDirectory):
    outputDirectory = outputDirectory + "//med-bayesian-plots"
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)
    name = os.path.splitext(os.path.basename(fileDir))[0]
    name = os.path.join(outputDirectory, name+'.csv')

    df = pd.read_csv(fileDir)
    if types == 1:
        new_c1 = [f'{shorten_name(x)}' for x in df['c1']]
        new_c2 = [f'{shorten_name(x)}' for x in df['c2']]
        df['c1'] = new_c1
        df['c2'] = new_c2
    elif types == 2:
        df.rename(columns=lambda x:shorten_name(x), inplace=True)
        df.rename(columns={'file':'File'}, inplace=True)
    else:
        raise Exception(f'Type: {type} is not valid')
    
    csv_data = df.to_csv(name, index=False)
    return name

def combine_probs_auc(probs, aucs, outputDirectory):
    outputDirectory = outputDirectory + "//med-bayesian-plots"
    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)
    name = os.path.splitext(os.path.basename(probs))[0]
    name = os.path.join(outputDirectory, name+'-med-bayes.csv')

    df_probs = pd.read_csv(probs)
    df_aucs = pd.read_csv(aucs)
    df_aucs = df_aucs.drop(columns=['File'])

    aucs_list = [{col:statistics.median(df_aucs[f'{col}'])} for col in df_aucs.columns]
    #convert list of dictionaries into single dictionary
    aucs_map = {k: v for d in aucs_list for k,v in d.items()}
    
    distribution_names = df_probs.c1.unique()
    new_distribution_names = [col for col in distribution_names]
    
    median_auc_list = [{dist: (statistics.median(df_probs[df_probs['c1'] == dist]['pwin']), statistics.median(df_probs[df_probs['c1'] == dist]['plose']))} for dist in distribution_names]
    median_auc_map = {k: v for d in median_auc_list for k,v in d.items()}

    output_df = pd.DataFrame()
    output_df['classifier'] = distribution_names
    output_df['pwin'] = [median_auc_map.get(dist)[0] for dist in new_distribution_names]
    output_df['plose'] = [median_auc_map.get(dist)[1] for dist in new_distribution_names]
    output_df['AUC'] = [aucs_map.get(dist) for dist in new_distribution_names]

    csv_data = output_df.to_csv(name, index=False)

def set_for_cd_diagram(fileDir, outputDirectory):
    df = pd.read_csv(fileDir)
    num_db = len(df)
    df = df.sort_values('File').reset_index(drop=True)
    outputDirectory = outputDirectory + "//cd-diagrams"
    name = os.path.splitext(os.path.basename(fileDir))[0]
    name = os.path.join(outputDirectory, name +'-critdiff.csv')

    if not os.path.exists(outputDirectory):
        print(f"Creating combined results directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    #results_out = open(name, "w+", newline='\n', encoding='utf-8')
    #results_out.write('classifier_name,dataset_name,AUC\n')

    datasets_names = df['File']
    output_df = pd.DataFrame()
    for col in df.columns:
        if col != 'File':
            cls_lst = [f'{shorten_name(col)}' for db in range(num_db)]
            temp_df = pd.DataFrame()
            temp_df['classifier_name'] = cls_lst
            temp_df['dataset_name'] = df['File']
            temp_df['AUC'] = df[f'{col}']
            output_df = pd.concat([output_df, temp_df])
    
    for col in output_df.columns:
        print(f"col: {col}")
    csv_data = output_df.to_csv(name, index = False)
        

def read_confusion_matrix(fileDir, fileName, outputDirectory):
    flag = False
    num_classes = 0
    confusion_matrix = []
    names = fileDir.split("-")
    with open(fileDir, "r") as f:
            for line in f:
                if line != '\n':
                    newline = line
                    if newline.split()[0] == 'Classes':
                        num_classes = int (newline.split()[1])
                    if newline.split()[0] == 'F1':
                        flag = True
                    elif flag:
                        matrix_line = newline.split()
                        matrix_line = [int(x) for x in matrix_line]
                        confusion_matrix.append(matrix_line)
            
    outputDirectory = outputDirectory + "\\confusion_matrix"

    if not os.path.exists(outputDirectory):
        print(f"Creating output directory: {outputDirectory}")
        os.makedirs(outputDirectory)

    name = os.path.splitext(os.path.basename(fileName))[0]
    name = os.path.join(outputDirectory, name+'-cf-matrix.csv')

    print()
    for i in range(len(confusion_matrix[0])):
        for j in range(len(confusion_matrix[0])):
            print(f"{confusion_matrix[i][j]} ", end='')
        print("")
    
    auc = obtainAUCMulticlass(confusion_matrix, num_classes)

    action = "Writing"
    if os.path.exists(name):
        action = "Appending"
        results_out = open(name, "a+", newline='\n', encoding='utf-8')
    else:
        results_out = open(name, "w", newline='\n', encoding='utf-8')
        results_out.write(f"File,AUC\n")

    results_out.write(f"{'-'.join(names[2:])},{str(auc)}\n")
    results_out.close()
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

def replace_rows(fileDir, newDir, replacement, output_directory, str_to_replace='Bhattacharyya'):
    df_comb = pd.read_csv(fileDir)
    df_comb = df_comb.sort_values('File').reset_index(drop=True)
    df_replacement = pd.read_csv(newDir)
    df_replacement = df_replacement.sort_values('File').reset_index(drop=True)
    df_comb = df_comb[~df_comb['eval_functions'].str.contains(str_to_replace)]
    df_comb = df_comb.append(df_replacement)
    df_comb = df_comb.sort_values('File').reset_index(drop=True)

    file_name = os.path.splitext(os.path.basename(fileDir))[0]
    file_name = os.path.join(output_directory, file_name+'-fixed.csv')

    df_comb.to_csv(file_name, index = False)




def pipeline(fileDir, originalDir, output_directory, k):
    order_file = order_results(fileDir, output_directory)
    order_auc, order_acc = separate(order_file, output_directory)
    auc_avg, acc_avg = average_k_runs_cross_validation(order_auc, k, output_directory)
    bayes_auc = leo_bayesian(auc_avg, output_directory)

def prepare_idv_files(fileDir, k, output_directory):
    order_file = order_results(fileDir, output_directory)
    order_auc, order_acc = separate(order_file, output_directory)
    auc_sort = sort_results(order_auc, output_directory)
    auc_avg, acc_avg = average_k_runs_cross_validation(auc_sort, k, output_directory)

def pipeline_leo(fileDir, originalDir, output_directory, k):
    shortend_bayes_auc = convert_names(fileDir, 1, output_directory)
    shortened_auc = convert_names(originalDir, 2, output_directory)
    combine_probs_auc(shortend_bayes_auc, shortened_auc, output_directory)

def pipeline_wilcoxon(fileDir, originalDir, output_directory, k):
    order_file = order_results(fileDir, output_directory)
    order_auc, order_acc = separate(order_file, output_directory)
    auc_avg, acc_avg = average_k_runs_cross_validation(order_auc, k, output_directory)
    transpose_auc_comb = append_results(auc_avg, originalDir, output_directory)
    wilcoxon_auc = wilcoxon(transpose_auc_comb, output_directory)
    analyze_wilcoxon(wilcoxon_auc, output_directory)

def pipeline_wilcoxon_cd(fileDir, output_directory):
    #This method begins from criticial difference file
    order_file = order_results(fileDir, output_directory)
    order_auc, order_acc = separate(order_file, output_directory)
    auc_comb = sort_results(order_auc, output_directory)
    wilcoxon_auc = wilcoxon(auc_comb, output_directory)
    analyze_wilcoxon(wilcoxon_auc, output_directory)

def pipeline_cd(fileDir, originalDir, output_directory, k):
    order_file = order_results(fileDir, output_directory)
    order_auc, order_acc = separate(order_file, output_directory)
    auc_avg, acc_avg = average_k_runs_cross_validation(order_auc, k, output_directory)
    auc_comb = append_results(auc_avg, originalDir, output_directory)
    set_for_cd_diagram(auc_comb, output_directory)

    
