import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import smallest_idx

class ReciprocalRankVoting:
    def __init__(self, evaluation_functions_names):
        self.reciprocal_rank = []
        self.reciprocal_vals = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)
        print(f"len of funcs: {len(self.evaluation_functions)}")   

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.reciprocal_vals.append(split_list)

    
    def get_best_split_idx(self):

        if len(self.reciprocal_vals) == 0:
            return
        self.reciprocal_vals = np.array(self.reciprocal_vals)
        self.reciprocal_vals = np.transpose(self.reciprocal_vals)

        self.reciprocal_vals = pd.DataFrame(self.reciprocal_vals)
        self.reciprocal_vals.columns = [f'CS{i}' for i,_ in enumerate(self.reciprocal_vals)]
        self.reciprocal_vals.index = [name for name in self.evaluation_functions]

        self.reciprocal_rank = self.reciprocal_vals.copy(deep=True)

        for index in self.reciprocal_rank.index:
            self.reciprocal_rank.loc[index] = self.reciprocal_rank.loc[index].rank(ascending=False, method='min', na_option='bottom')
        
        irp = self.inverse_rank_position()        
        #reset for future cycles
        self.reciprocal_rank = []
        self.reciprocal_vals = []
        #print(f"irp:{irp}")
        return irp
    
    def inverse_rank_position(self):
        rank_lst = [sum(1/x for x in self.reciprocal_rank[col])**-1 for col in self.reciprocal_rank]
        #print(f"rank_lst:{rank_lst}")
        return smallest_idx(rank_lst)

    