import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence, MultiClassBhattacharyya
from .EvaluationFunctionCombinerHelper import get_functions_dict
from .Helpers import largest_idx, get_smallest_val

class SchulzeVoting:
    def __init__(self, evaluation_functions_names):
        self.table_rank = []
        self.table_vals = []
        self.preferences = []
        self.evaluation_functions = get_functions_dict(evaluation_functions_names)

    def add_candidate_splits(self, parent, children):
        split_list = []
        for func in self.evaluation_functions.values():
            split_list.append(func(parent, children))
        self.table_vals.append(split_list)

    def get_best_split_idx(self):
        if len(self.table_vals) == 0:
            return

        self.get_preferences()

        p = self.get_strongest_paths()
        df_p = pd.DataFrame(p)

        num_candidates = len(df_p)
        victory_lst = [0] * num_candidates

        for i in range(num_candidates):
            for j in range(num_candidates):
                if df_p[i][j] > df_p[j][i]:
                    victory_lst[i] = victory_lst[i]+1
        best_idx = largest_idx(victory_lst)
        return best_idx
    
    def get_preferences(self):

        self.table_vals = np.array(self.table_vals)
        self.table_vals = np.transpose(self.table_vals)

        self.table_vals = pd.DataFrame(self.table_vals)
                
        self.table_rank = self.table_vals.copy(deep=True)

        self.preferences = [[0]*len(self.table_rank.columns) for i in range(len(self.table_rank.columns))]
        self.preferences = pd.DataFrame(self.preferences)

        for coli in range(len(self.table_rank)):
            for colj in self.table_rank:
                temp =( self.table_rank.loc[coli][colj] > self.table_rank.loc[coli]).astype(int)
                self.preferences.loc[colj] = self.preferences.loc[colj] + temp
        
    
    def get_strongest_paths(self):
        p = [[0]*len(self.preferences) for i in range(len(self.preferences))]
        for i in range(len(self.preferences)):
            for j in range(len(self.preferences)):
                if i != j:
                    if self.preferences[i][j] > self.preferences[j][i]:
                        p[i][j] = self.preferences[i][j]
                    else:
                        p[i][j] = 0
        for i in range(len(self.preferences)):
            for j in range(len(self.preferences)):
                if i != j:
                    for k in range(len(self.preferences)):
                        if i != k and j != k:
                            p[j][k] = max(p[j][k], min(p[j][i],p[i][k]))
        return p

    