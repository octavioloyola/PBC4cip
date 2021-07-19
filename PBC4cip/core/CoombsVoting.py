import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import largest_idx, get_largest_val

class CoombsVoting:
    def __init__(self, evaluation_functions_names):
        self.coombs_table_rank = []
        self.coombs_table_vals = []
        self.coombs_table_rank_max = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.coombs_table_vals.append(split_list)

    def get_best_split_idx(self):
        if len(self.coombs_table_vals) == 0:
            return
        self.coombs_table_vals = np.array(self.coombs_table_vals)
        self.coombs_table_vals = np.transpose(self.coombs_table_vals)

        self.coombs_table_vals = pd.DataFrame(self.coombs_table_vals)
        self.coombs_table_vals.columns = [f'CS{i}' for i,_ in enumerate(self.coombs_table_vals)]
        self.coombs_table_vals.index = [name for name in self.evaluation_functions]
        
        self.coombs_table_rank = self.coombs_table_vals.copy(deep=True)
        self.coombs_table_rank_max = self.coombs_table_vals.copy(deep=True)
        for index in self.coombs_table_rank.index:
            self.coombs_table_rank.loc[index] = self.coombs_table_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        one_rows = self.coombs_table_rank[self.coombs_table_rank==1].count()
        one_sums = self.coombs_table_rank.eq(1).sum()

        for index in self.coombs_table_rank_max.index:
                self.coombs_table_rank_max.loc[index] = self.coombs_table_vals.loc[index].rank(ascending=False,method='max', na_option='bottom')
        last_place_rows = self.coombs_table_rank_max[self.coombs_table_rank_max == len(self.coombs_table_rank_max.columns)].count()
        while not((one_rows[0] == one_rows).all()) and not ((last_place_rows[0] == last_place_rows).all()) and not any(one_sums > len(self.coombs_table_rank) // 2):
            print(f"table:\n{self.coombs_table_vals}")
            self.drop_last_place()

            for index in self.coombs_table_rank.index:
                self.coombs_table_rank.loc[index] = self.coombs_table_vals.loc[index].rank(ascending=False,method='min', na_option='bottom')

            for index in self.coombs_table_rank.index:
                self.coombs_table_rank_max.loc[index] = self.coombs_table_vals.loc[index].rank(ascending=False,method='max', na_option='bottom')
            one_rows = self.coombs_table_rank[self.coombs_table_rank==1].count()
            one_sums = self.coombs_table_rank.eq(1).sum()

            last_place_rows = self.coombs_table_rank_max[self.coombs_table_rank_max == len(self.coombs_table_rank_max.columns)].count()
        
        best_idx = int ((self.coombs_table_rank.eq(1).sum() > len(self.coombs_table_rank) // 2).idxmax().replace('CS',''))
        #reset for future cycles    
        self.coombs_table_vals = []
        self.coombs_table_rank = []
        self.coombs_table_rank_max = []
        return best_idx
    
    def drop_last_place(self):
        
        cols = self.coombs_table_rank_max.eq(len(self.coombs_table_rank_max.columns)).sum()
        last_place = get_largest_val(cols)
        cols_to_delete = cols[cols == last_place]
        self.coombs_table_rank.drop(cols_to_delete.index, axis=1, inplace=True)
        self.coombs_table_vals.drop(cols_to_delete.index, axis=1, inplace=True)
        self.coombs_table_rank_max.drop(cols_to_delete.index, axis=1, inplace=True)

    