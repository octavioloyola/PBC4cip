import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .Helpers import largest_idx, get_smallest_val

class InstantRunoffVoting:
    def __init__(self, evaluation_functions_names):
        self.irv_table_rank = []
        self.irv_table_vals = []
        self.evaluation_functions = self.get_functions_dict(evaluation_functions_names)
        #print(f"len of funcs: {len(self.evaluation_functions)}")   

    def irv(self, parent, children):
        #print(f"init!!!\n\n!!!\n!!")
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.irv_table_vals.append(split_list)

    def irv_evaluate(self):
        if len(self.irv_table_vals) == 0:
            return
        self.irv_table_vals = np.array(self.irv_table_vals)
        self.irv_table_vals = np.transpose(self.irv_table_vals)

        self.irv_table_vals = pd.DataFrame(self.irv_table_vals)
        self.irv_table_vals.columns = [f'CS{i}' for i,_ in enumerate(self.irv_table_vals)]
        self.irv_table_vals.index = [name for name in self.evaluation_functions]
        
        self.irv_table_rank = self.irv_table_vals.copy(deep=True)
        #print(f"lenTable: {len(self.irv_table_rank.columns)}")
        for index in self.irv_table_rank.index:
            #self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False,method='min', na_option='bottom')
            self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        one_rows = self.irv_table_rank[self.irv_table_rank==1].count()
        one_sums = self.irv_table_rank.eq(1).sum()
        
        while not((one_rows[0] == one_rows).all()) and not any(one_sums > len(self.irv_table_rank) // 2):
            #print(f"Pog:\n{self.irv_table_rank[self.irv_table_rank==1].count()[0] == self.irv_table_rank[self.irv_table_rank == 1].count().all()}")
            #print(f"aaaa:\n{not( (self.irv_table_rank[self.irv_table_rank==1].count()[0] == self.irv_table_rank[self.irv_table_rank == 1].count()).all())}")
            #print(f"bbbb:\n{not any((self.irv_table_rank.eq(1).sum() > len(self.irv_table_rank.dropna()) // 2))}")
            #print(f"lenTablee: {len(self.irv_table_rank.columns)}")
            #print(f"table:\n{self.irv_table_vals}")
            #print(f"pre good:\n {self.irv_table_rank}")
            self.drop_last_place()
            #print(f"len: {len(self.irv_table_rank.columns)}")

            #self.irv_table_rank = self.irv_table_vals.copy(deep=True)
            for index in self.irv_table_rank.index:
                self.irv_table_rank.loc[index] = self.irv_table_vals.loc[index].rank(ascending=False,method='min', na_option='bottom')
                #self.irv_table_rank.loc[index] = self.irv_table_vals.loc[index].rank(ascending=False, method='min')
            #print(f"good table:\n{self.irv_table_rank}")
            one_rows = self.irv_table_rank[self.irv_table_rank==1].count()
            one_sums = self.irv_table_rank.eq(1).sum()
            #print(f"lennDrop_ {len(self.irv_table_rank.columns)}")
        
        #print(f"last_table:\n{self.irv_table_rank}")
        best_idx = int ((self.irv_table_rank.eq(1).sum() > len(self.irv_table_rank) // 2).idxmax().replace('CS',''))
        #reset for future cycles    
        self.irv_table_vals = []
        self.irv_table_rank = []
        #print(f"best_idx is: {best_idx}")
        return best_idx
    
    def drop_last_place(self):
        #rank_lst = [self.irv_table_rank[f'{col}'].sum() for col in list(self.irv_table_rank) ]
        #print(f"rank lst:\n{rank_lst}")
        #worst_idx = largest_idx(rank_lst)
        #print(f"worst place: {worst_idx}")
        #print(f"aaa: {self.irv_table_rank.columns[worst_idx]}")
        #self.irv_table_vals.drop(self.irv_table_vals.columns[worst_idx], axis=1, inplace=True)
        #self.irv_table_rank.drop(self.irv_table_rank.columns[worst_idx], axis=1, inplace=True)
        cols = self.irv_table_rank.eq(1.0).sum()
        #print(f"cols:\n{cols}")
        last_place = get_smallest_val(cols)
        cols_to_delete = cols[cols == last_place]
        self.irv_table_rank.drop(cols_to_delete.index, axis=1, inplace=True)
        self.irv_table_vals.drop(cols_to_delete.index, axis=1, inplace=True)



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

    