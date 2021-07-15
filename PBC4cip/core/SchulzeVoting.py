import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .Helpers import largest_idx

class Schulze:
    def __init__(self, evaluation_functions_names):
        self.irv_table_rank = []
        self.irv_table_vals = []
        self.evaluation_functions = self.get_functions_dict(evaluation_functions_names)
        #print(f"len of funcs: {len(self.evaluation_functions)}")   

    def schulze(self, parent, children):
        #print(f"init!!!\n\n!!!\n!!")
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.irv_table_vals.append(split_list)

    def schulze_evaluate(self):
        if len(self.irv_table_vals) == 0:
            return
        self.irv_table_vals = np.array(self.irv_table_vals)
        print(f"pog?")
        self.irv_table_vals = np.transpose(self.irv_table_vals)

        self.irv_table_vals = pd.DataFrame(self.irv_table_vals)
        self.irv_table_vals.columns = [f'CS{i}' for i,_ in enumerate(self.irv_table_vals)]
        self.irv_table_vals.index = [name for name in self.evaluation_functions]
        
        self.irv_table_rank = self.irv_table_vals.copy(deep=True)
        print(f"lenTable: {len(self.irv_table_rank.columns)}")
        for index in self.irv_table_rank.index:
            #self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False,method='min', na_option='bottom')
            self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False, method='min')
        #realLen = len(self.irv_table_rank.dropna())
        
        while not any((self.irv_table_rank.eq(1).sum() > len(self.irv_table_rank.dropna()) // 2)):
            print(f"lenTablee: {len(self.irv_table_rank.columns)}")
            #print(f"table:\n{self.irv_table_vals}")
            #print(f"pre good:\n {self.irv_table_rank}")
            #print(f"len: {len(self.irv_table_rank.columns)}")

            #self.irv_table_rank = self.irv_table_vals.copy(deep=True)
            for index in self.irv_table_rank.index:
                #self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False,method='min', na_option='bottom')
                self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False, method='min')
            #print(f"good table:\n{self.irv_table_rank}")
            if len(self.irv_table_rank.columns) == 1:
                break
            self.drop_last_place()
            print(f"lennDrop_ {len(self.irv_table_rank.columns)}")
        
        #print(f"about to get best")
        best_idx = int ((self.irv_table_rank.eq(1).sum() > len(self.irv_table_rank.dropna()) // 2).idxmax().replace('CS',''))
        #reset for future cycles    
        self.irv_table_vals = []
        self.irv_table_rank = []
        #print(f"best_idx is: {best_idx}")
        return best_idx

    def is_majority(self):
        if len(self.irv_table_rank) <= 0:
            return False
        
        realLen = len(self.irv_table_rank.dropna())
        #print(f"realLen: {realLen}")
        num_candidate_splits = len(self.irv_table_rank.columns)
        for col in self.irv_table_rank.columns:
            amount_fc_votes = len(self.irv_table_rank[self.irv_table_rank[col] == 1.0])
            #print(f"fc: {amount_fc_votes}")
            if amount_fc_votes > realLen / 2:
                #print(f"return True")
                return True

       #print(f"return False")
        return False
    
    def get_majority(self):
        if len(self.irv_table_rank) <= 0:
            raise Exception(f"irv_table must have at least 1 element")
        
        num_candidate_splits = len(self.irv_table_rank.columns)
        realLen = len(self.irv_table_rank.dropna())
        #print(f"realLen:{realLen}")
        for col in self.irv_table_rank.columns:
            amount_fc_votes = len(self.irv_table_rank[self.irv_table_rank[col] == 1.0])
            #print(f"fc get: {amount_fc_votes}")
            if amount_fc_votes > realLen / 2:
                #print(f"return majority")
                return int(col.replace('CS',''))
        
        #print(f"Check Emergency Case:\n{self.irv_table_rank}")
        if len(self.irv_table_rank.columns) == 2:
            return int(self.irv_table_rank.columns[0].replace('CS',''))
    
    def drop_last_place(self):
        rank_lst = [self.irv_table_rank[f'{col}'].sum() for col in list(self.irv_table_rank) ]
        #print(f"rank lst:\n{rank_lst}")
        worst_idx = largest_idx(rank_lst)
        #print(f"worst place: {worst_idx}")
        print(f"aaa: {self.irv_table_rank.columns[worst_idx]}")
        self.irv_table_vals.drop(self.irv_table_vals.columns[worst_idx], axis=1, inplace=True)
        self.irv_table_rank.drop(self.irv_table_rank.columns[worst_idx], axis=1, inplace=True)

    def get_functions_dict(self, func_names):
        func_names = [name.lower() for name in func_names]
        evaluator_dict = {
                'twoing': Twoing,
                'quinlan gain': QuinlanGain,
                'gini impurity': GiniImpurity,
                'multi class hellinger': MultiClassHellinger,
                'chi squared': ChiSquared,
                'dkm': DKM,
                'g statistic': G_Statistic,
                'marsh': MARSH,
                'normalized gain': NormalizedGain,
                'kolmogorov': KolmogorovDependence,
                'bhattacharyya': MultiClassBhattacharyya
            }
        return {key:value for (key,value) in evaluator_dict.items() if key in func_names}

    