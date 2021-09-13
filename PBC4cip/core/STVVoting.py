import numpy as np
import pandas as pd
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import smallest_random_idx

class STVVoting:
    def __init__(self, evaluation_functions_names):
        self.stv_table_rank = []
        self.stv_table_vals = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.stv_table_vals.append(split_list)

    def get_best_split_idx(self):
        if len(self.stv_table_vals) == 0:
            return
        self.stv_table_vals = np.array(self.stv_table_vals)
        self.stv_table_vals = np.transpose(self.stv_table_vals)

        self.stv_table_vals = pd.DataFrame(self.stv_table_vals)
        self.stv_table_vals.columns = [f'CS{i}' for i,_ in enumerate(self.stv_table_vals)]
        self.stv_table_vals.index = [name for name in self.evaluation_functions]
        
        self.stv_table_rank = self.stv_table_vals.copy(deep=True)
        for index in self.stv_table_rank.index:
            self.stv_table_rank.loc[index] = self.stv_table_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        self.drop_no_vote_candidates()
        one_sums = self.stv_table_rank.eq(1).sum()
        
        
        while not any(one_sums > len(self.stv_table_rank) // 2):
            self.drop_last_place()

            for index in self.stv_table_rank.index:
                self.stv_table_rank.loc[index] = self.stv_table_vals.loc[index].rank(ascending=False,method='min', na_option='bottom')
            one_sums = self.stv_table_rank.eq(1).sum()
        
        best_idx = int ((self.stv_table_rank.eq(1).sum() > len(self.stv_table_rank) // 2).idxmax().replace('CS',''))
        #reset for future cycles    
        self.stv_table_vals = []
        self.stv_table_rank = []
        return best_idx
    
    def drop_last_place(self):
        cols = self.stv_table_rank.eq(1.0).sum()
        last_place = smallest_random_idx(cols)
        self.stv_table_rank.drop(self.stv_table_rank.columns[last_place], axis=1, inplace=True)
        self.stv_table_vals.drop(self.stv_table_vals.columns[last_place], axis=1, inplace=True)
    
    def drop_no_vote_candidates(self):
        cols = self.stv_table_rank.eq(1.0).sum()==0
        cols_to_delete = cols[cols==True]
        self.stv_table_rank.drop(cols_to_delete.index, axis=1, inplace=True)
        self.stv_table_vals.drop(cols_to_delete.index, axis=1, inplace=True)
    