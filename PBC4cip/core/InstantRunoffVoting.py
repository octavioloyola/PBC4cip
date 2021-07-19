import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import largest_idx, get_smallest_val

class InstantRunoffVoting:
    def __init__(self, evaluation_functions_names):
        self.irv_table_rank = []
        self.irv_table_vals = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.irv_table_vals.append(split_list)

    def get_best_split_idx(self):
        if len(self.irv_table_vals) == 0:
            return
        self.irv_table_vals = np.array(self.irv_table_vals)
        self.irv_table_vals = np.transpose(self.irv_table_vals)

        self.irv_table_vals = pd.DataFrame(self.irv_table_vals)
        self.irv_table_vals.columns = [f'CS{i}' for i,_ in enumerate(self.irv_table_vals)]
        self.irv_table_vals.index = [name for name in self.evaluation_functions]
        
        self.irv_table_rank = self.irv_table_vals.copy(deep=True)
        for index in self.irv_table_rank.index:
            self.irv_table_rank.loc[index] = self.irv_table_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        one_rows = self.irv_table_rank[self.irv_table_rank==1].count()
        one_sums = self.irv_table_rank.eq(1).sum()
        
        while not((one_rows[0] == one_rows).all()) and not any(one_sums > len(self.irv_table_rank) // 2):
            self.drop_last_place()

            for index in self.irv_table_rank.index:
                self.irv_table_rank.loc[index] = self.irv_table_vals.loc[index].rank(ascending=False,method='min', na_option='bottom')
            one_rows = self.irv_table_rank[self.irv_table_rank==1].count()
            one_sums = self.irv_table_rank.eq(1).sum()
        
        best_idx = int ((self.irv_table_rank.eq(1).sum() > len(self.irv_table_rank) // 2).idxmax().replace('CS',''))
        #reset for future cycles    
        self.irv_table_vals = []
        self.irv_table_rank = []
        return best_idx
    
    def drop_last_place(self):
        cols = self.irv_table_rank.eq(1.0).sum()
        last_place = get_smallest_val(cols)
        cols_to_delete = cols[cols == last_place]
        self.irv_table_rank.drop(cols_to_delete.index, axis=1, inplace=True)
        self.irv_table_vals.drop(cols_to_delete.index, axis=1, inplace=True)

    