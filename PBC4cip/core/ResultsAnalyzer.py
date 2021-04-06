import os
import csv
import math
from tqdm import tqdm
from scipy import stats
import pandas as pd
import numpy as np

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
        print(f"Creating stat test directory: {output_directory}")
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


def wilcoxon_old(fileDir, output_directory):
    df = pd.read_csv(fileDir)
    auc_df = df.filter(regex='-AUC')
    acc_df = df.filter(regex='-Acc')
    #auc_df['File'] = df['File']
    #acc_df['File'] = df['File']
    #print(auc_df.columns)
    #print(auc_df)
    #print(acc_df)
    auc_lst = []
    W_pos_auc_lst = []
    W_neg_auc_lst = []
    for col_x in auc_df.columns:
        for col_y in auc_df.columns:
            if col_x != col_y:
                w, p = stats.wilcoxon(auc_df[f'{col_x}'], auc_df[f'{col_y}'])
                auc_lst.append(p)
                auc_diff = [auc_df[f'{col_x}'][i] - auc_df[f'{col_y}'][i] for i in range(len(auc_df[f'{col_x}']))]
                auc_sign_idx_pos = [i for i in range(len(auc_diff)) if auc_diff[i] > 0]
                auc_sign_idx_neg = [i for i in range(len(auc_diff)) if auc_diff[i] < 0]
                auc_abs_diff = [abs(x) for x in auc_diff]
                auc_ranks = stats.rankdata(auc_abs_diff, method='dense')
                W_pos = sum([auc_ranks[i] for i in range(len(auc_ranks)) if i in auc_sign_idx_pos ])
                W_pos_auc_lst.append(W_pos)
                W_neg = sum([auc_ranks[i] for i in range(len(auc_ranks)) if i in auc_sign_idx_neg ])
                W_neg_auc_lst.append(W_neg)
    
    acc_lst = []
    W_pos_acc_lst = []
    W_neg_acc_lst = []

    for col_x in acc_df.columns:
        for col_y in acc_df.columns:
            if col_x != col_y:
                w, p = stats.wilcoxon(acc_df[f'{col_x}'], acc_df[f'{col_y}'])
                acc_lst.append(p)
                acc_diff = [acc_df[f'{col_x}'][i] - acc_df[f'{col_y}'][i] for i in range(len(acc_df[f'{col_x}']))]
                acc_sign_idx_pos = [i for i in range(len(acc_diff)) if acc_diff[i] > 0]
                acc_sign_idx_neg = [i for i in range(len(acc_diff)) if acc_diff[i] < 0]
                acc_abs_diff = [abs(x) for x in acc_diff]
                acc_ranks = stats.rankdata(acc_abs_diff)
                W_pos = sum([acc_ranks[i] for i in range(len(acc_ranks)) if i in acc_sign_idx_pos ])
                W_pos_acc_lst.append(W_pos)
                W_neg = sum([acc_ranks[i] for i in range(len(acc_ranks)) if i in acc_sign_idx_neg ])
                W_neg_acc_lst.append(W_neg)


                #acc_diff = [acc_df[f'{col_x}'][i] - acc_df[f'{col_y}'][i] for i in range(len(acc_df[f'{col_x}']))]
    #print(stats.wilcoxon(auc_df['QuinlanGain-G-Statistic-AUC'], auc_df['QuinlanGain-MCH-AUC']))
    print(auc_lst)
    print(len(auc_lst))
    print(W_pos_auc_lst)
    print(W_neg_auc_lst)
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
                auc_results_out.write(f"{combination},{str(auc_lst[idx])},{str(W_pos_auc_lst[idx])},{str(W_neg_auc_lst[idx])}\n")
                idx = idx+1
        auc_results_out.write(f"\n")

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
                combination = col_x[0:len(col_x)-4] + " vs " + col_y[0:len(col_y)-4]
                acc_results_out.write(f"{combination},{str(acc_lst[idx])},{str(W_pos_acc_lst[idx])},{str(W_neg_acc_lst[idx])}\n")
                idx = idx+1
        acc_results_out.write(f"\n")

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