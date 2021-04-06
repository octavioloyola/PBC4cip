import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .Helpers import smallest_idx

class EvaluationFunctionCombiner:
    def __init__(self, evaluation_functions_names):
        self.borda_count_table = []
        self.evaluation_functions = self.get_functions_dict(evaluation_functions_names)   

    def borda_count(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.borda_count_table.append(split_list)

    def borda_count_evaluate(self):
        if len(self.borda_count_table) == 0:
            return
        self.borda_count_table = np.array(self.borda_count_table)
        self.borda_count_table = np.transpose(self.borda_count_table)

        self.borda_count_table = pd.DataFrame(self.borda_count_table)
        self.borda_count_table.columns = [f'CS{i}' for i,_ in enumerate(self.borda_count_table)]
        self.borda_count_table.index = [name for name in self.evaluation_functions]

        #print(self.borda_count_table)
    
        for index in self.borda_count_table.index:
            self.borda_count_table.loc[index] = self.borda_count_table.loc[index].rank(ascending=False)
        rank_lst = [self.borda_count_table[f'{col}'].sum() for col in list(self.borda_count_table) ]
        best_idx = smallest_idx(rank_lst)

        self.borda_count_table = [] #reset for future cycles
        return smallest_idx(rank_lst)


    def get_functions_dict(self, func_names):
        func_names = [name.lower() for name in func_names]
        evaluator_dict = {
                'twoing': Twoing,
                'quinlan gain': QuinlanGain,
                'gini impurity': GiniImpurity,
                'multi class hellinger': MultiClassHellinger,
                'chi-squared': ChiSquared,
                'dkm': DKM,
                'g-statistic': G_Statistic,
                'marsh': MARSH,
                'normalized gain': NormalizedGain,
                'kolmogorov': KolmogorovDependence,
                'bhattacharyya': MultiClassBhattacharyya
            }
        return {key:value for (key,value) in evaluator_dict.items() if key in func_names}

    