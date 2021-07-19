import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import largest_idx, get_smallest_val

class BucklinVoting:
    def __init__(self, evaluation_functions_names):
        self.bucklin_table_rank = []
        self.bucklin_table_vals = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.bucklin_table_vals.append(split_list)

    def get_best_split_idx(self):
        if len(self.bucklin_table_vals) == 0:
            return
        self.bucklin_table_vals = np.array(self.bucklin_table_vals)
        self.bucklin_table_vals = np.transpose(self.bucklin_table_vals)

        self.bucklin_table_vals = pd.DataFrame(self.bucklin_table_vals)
        self.bucklin_table_vals.index = [name for name in self.evaluation_functions]
        
        self.bucklin_table_rank = self.bucklin_table_vals.copy(deep=True)
        for index in self.bucklin_table_rank.index:
            self.bucklin_table_rank.loc[index] = self.bucklin_table_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        
        self.bucklin_sums = [0] * len(self.bucklin_table_rank.columns)
        self.bucklin_sums = pd.Series(self.bucklin_sums)
        idx_sums = self.bucklin_table_rank.eq(1).sum()

        self.bucklin_sums = self.bucklin_sums + idx_sums

        sum_idx = 1
        while not any(self.bucklin_sums > len(self.bucklin_table_rank) // 2):
            sum_idx = sum_idx + 1

            idx_sums = self.bucklin_table_rank.eq(sum_idx).sum()
            self.bucklin_sums = self.bucklin_sums + idx_sums
        
        best_idx = largest_idx(self.bucklin_sums)
        #reset for future cycles    
        self.bucklin_table_vals = []
        self.bucklin_table_rank = []
        return best_idx
    

    