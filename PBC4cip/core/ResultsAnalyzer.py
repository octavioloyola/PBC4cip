import csv
import math
import os
import random

import baycomp
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def show_results(confusion, acc, auc, numPatterns):
    print()
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            print(f"{confusion[i][j]} ", end='')
        print("")
    print(f"acc: {acc} , auc: {auc} , numPatterns: {numPatterns}")

def order_results(fileDir, column_names, output_directory):
    df = pd.read_csv(fileDir)
    with open(column_names, "r") as f:
        col_names = f.readlines()
        col_names = [line.replace("\n", "") for line in col_names]
    
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
    
    results_out = open(file_name, "a", newline='\n', encoding='utf-8')
    results_out.write(f"File,AUC,Acc,distribution-evaluator\n")

    for name in col_names:
        for i,row in df.iterrows():
            #print(f"name:{name}")
            #if row['distribution_evaluator'] == 'combiner':
            #    print(f"dist: {row['eval_functions']}")
            if str(row['distribution_evaluator']).strip() == name or str(row['eval_functions']).strip() == name:
                if str(row['distribution_evaluator']).strip() == 'combiner':
                    results_out.write(f"{str(row['File'])},{str(row['AUC'])},{str(row['ACC'])},{str(row['eval_functions'])}\n")
                else:
                    results_out.write(f"{str(row['File'])},{str(row['AUC'])},{str(row['ACC'])},{str(row['distribution_evaluator'])}\n")
    results_out.close()
    transpose_results(file_name, column_names, output_directory)

def transpose_results(fileDir, column_names, output_directory):
    df = pd.read_csv(fileDir)
    with open(column_names, "r") as f:
        col_names = f.readlines()
        col_names = [line.replace("\n", "") for line in col_names]
    
    print(f"col_names: {len(col_names)}")
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
        #print(f"idx:{lst}")
        #print(f"result:{result}\n\n")
        results_out.write(str(df.at[i,'File'])+','+",".join(result)+"\n")
    results_out.close()


def wilcoxon(fileDir, output_directory):
    df = pd.read_csv(fileDir)
    auc_df = df.filter(regex='-AUC')
    acc_df = df.filter(regex='-Acc')

    auc_lst = []
    auc_w_one = []
    auc_w_two = []
    for col_x in auc_df.columns:
        for col_y in auc_df.columns:
            if col_x != col_y:
                #print(f"{col_x} and {col_y}")
                x,y = map(np.asarray, (auc_df[f'{col_x}'],auc_df[f'{col_y}']))
                d = x-y
                d = np.compress(np.not_equal(d, 0), d)

                r = stats.rankdata(abs(d))
                r_plus = np.sum((d > 0)* r)
                r_minus = np.sum((d < 0) * r)
                auc_w_one.append(r_plus)
                auc_w_two.append(r_minus)

                w,p = stats.wilcoxon(x,y)
                auc_lst.append(p)

    acc_lst = []
    acc_w_one = []
    acc_w_two = []
    for col_x in acc_df.columns:
        for col_y in acc_df.columns:
            if col_x != col_y:
                #print(f"{col_x} and {col_y}")
                x,y = map(np.asarray, (acc_df[f'{col_x}'],acc_df[f'{col_y}']))
                d = x-y
                d = np.compress(np.not_equal(d, 0), d)

                r = stats.rankdata(abs(d))
                r_plus = np.sum((d > 0)* r)
                r_minus = np.sum((d < 0) * r)
                acc_w_one.append(r_plus)
                acc_w_two.append(r_minus)

                w,p = stats.wilcoxon(x,y)
                acc_lst.append(p)
    
    output_directory = output_directory + "\\stat-tests"

    if not os.path.exists(output_directory):
        print(f"Creating stat test directory: {output_directory}")
        os.makedirs(output_directory)
        
    auc_name = os.path.splitext(os.path.basename(fileDir))[0]
    auc_name = os.path.join(output_directory, auc_name +'-auc.csv')

    action = "Writing"
    if os.path.exists(auc_name):
        action = "Overwriting"
        os.remove(auc_name)
    
    auc_results_out = open(auc_name, "a", newline='\n', encoding='utf-8')
    auc_results_out.write(f"Combination,P-Value,W-Pos,W-Neg\n")
    idx = 0
    for i, col_x in enumerate(auc_df.columns):
        for j, col_y in enumerate(auc_df.columns):
            if col_x != col_y:
                combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
                auc_results_out.write(f"{combination},{str(auc_lst[idx])},{str(auc_w_one[idx])},{str(auc_w_two[idx])}\n")
                idx = idx+1
        auc_results_out.write(f"\n")
    auc_results_out.close()
    
    acc_name = os.path.splitext(os.path.basename(fileDir))[0]
    acc_name = os.path.join(output_directory, acc_name+'-acc.csv')

    action = "Writing"
    if os.path.exists(acc_name):
        action = "Overwriting"
        os.remove(acc_name)
    
    acc_results_out = open(acc_name, "a", newline='\n', encoding='utf-8')
    acc_results_out.write(f"Combination,P-Value,W-Pos,W-Neg\n")
    idx = 0
    for i, col_x in enumerate(acc_df.columns):
        for j, col_y in enumerate(acc_df.columns):
            if col_x != col_y:
                combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
                acc_results_out.write(f"{combination},{str(acc_lst[idx])},{str(acc_w_one[idx])},{str(acc_w_two[idx])}\n")
                idx = idx+1
        acc_results_out.write(f"\n")

    acc_results_out.close()

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
    #iterations = df.shape[0] // k
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

    p_left_lst = [[0]*len(num_df.columns) for i in enumerate(num_df.columns)]
    p_rope_lst = [[0]*len(num_df.columns) for i in enumerate(num_df.columns)]
    p_right_lst = [[0]*len(num_df.columns) for i in enumerate(num_df.columns)]

    output_directory = output_directory + "\\bayesian-tests"

    if not os.path.exists(output_directory):
        print(f"Creating bayesian test directory: {output_directory}")
        os.makedirs(output_directory)
        
    result_name = os.path.splitext(os.path.basename(fileDir))[0]
    print(f"len: {len(os.path.splitext(os.path.basename(fileDir)))}")
    print(f"{os.path.splitext(os.path.basename(fileDir))}")
    result_name = os.path.join(output_directory, result_name +'-bayes.csv')

    action = "Writing"
    if os.path.exists(result_name):
        action = "Overwriting"
        os.remove(result_name)
    
    results_out = open(result_name, "a", newline='\n', encoding='utf-8')
    results_out.write(f"Combination,P-Left,P-ROPE,P-Right\n")

    ix = 0
    iy = 0
    for col_x in tqdm(num_df.columns, desc=f"Performing bayesian analysis...", unit="col_x", leave=False):
        for col_y in tqdm(num_df.columns, desc=f"vs {col_x}...", unit="col_y", leave=False):
            combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
            if col_x != col_y and ix >= iy:
                x,y = map(np.asarray, (num_df[f'{col_x}'],num_df[f'{col_y}']))
                left,rope,right = baycomp.two_on_multiple(x,y, rope = 0.01, runs=runs)
                #left,rope,right = ((random.randint(1,10),random.randint(1,10),random.randint(1,10)))
                p_left_lst[ix][iy] = left 
                p_rope_lst[ix][iy] = rope
                p_right_lst[ix][iy] = right

                results_out.write(f"{combination},{str(left)},{str(rope)},{str(right)}\n")
            elif col_x != col_y:
                results_out.write(f"{combination},{str(p_right_lst[iy][ix])},{str(p_rope_lst[iy][ix])},{str(p_left_lst[iy][ix])}\n")
            ix = ix+1
        ix = 0    
        iy = iy+1
        results_out.write(f"\n")
    results_out.close() 


    """
    idx = 0
    for i, col_x in enumerate(num_df.columns):
        for j, col_y in enumerate(num_df.columns):
            if col_x != col_y:
                combination = col_x[0:len(col_x)-4] + " vs " +  col_y[0:len(col_y)-4]
                results_out.write(f"{combination},{str(p_left_lst[idx])},{str(p_rope_lst[idx])},{str(p_right_lst[idx])}\n")
                idx = idx+1
        results_out.write(f"\n")
    results_out.close()
    """


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
    
    csv_data = auc_df.to_csv(auc_name, index = False)

def average_k_runs_cross_validation(fileDir, k, output_directory):
    df = pd.read_csv(fileDir)
    chunk_size = len(df['File']) // k
    auc_df = df.filter(regex='-AUC')
    acc_df = df.filter(regex='-Acc')

    #print(auc_df.head())

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
    auc_name = os.path.join(output_directory, auc_name +'-auc-avg.csv')

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
    acc_name = os.path.join(output_directory, acc_name +'-acc-avg.csv')

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

def read_shdz_results(fileDir, fileName, outputDirectory):
    auc = None
    acc = None
    #print(f"fileDir: {fileName}")
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
    name = os.path.join(outputDirectory, name+'-ordered.csv')

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
