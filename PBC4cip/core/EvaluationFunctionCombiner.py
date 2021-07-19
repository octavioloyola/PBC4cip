import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import smallest_idx

class EvaluationFunctionCombiner:
    def __init__(self, evaluation_functions_names):
        self.borda_count_table = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)
        print(f"len of funcs: {len(self.evaluation_functions)}")   

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.borda_count_table.append(split_list)

    def get_best_split_idx(self):
        if len(self.borda_count_table) == 0:
            return
        self.borda_count_table = np.array(self.borda_count_table)
        self.borda_count_table = np.transpose(self.borda_count_table)

        self.borda_count_table = pd.DataFrame(self.borda_count_table)
        self.borda_count_table.columns = [f'CS{i}' for i,_ in enumerate(self.borda_count_table)]
        self.borda_count_table.index = [name for name in self.evaluation_functions]
    
        for index in self.borda_count_table.index:
            self.borda_count_table.loc[index] = self.borda_count_table.loc[index].rank(ascending=False)
        rank_lst = [self.borda_count_table[f'{col}'].sum() for col in list(self.borda_count_table) ]
        best_idx = smallest_idx(rank_lst)

        self.borda_count_table = [] #reset for future cycles
        return smallest_idx(rank_lst)


    