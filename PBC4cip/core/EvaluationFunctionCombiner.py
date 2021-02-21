import numpy as np
import pandas as pd
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence
from .Helpers import smallest_idx

class EvaluationFunctionCombiner():
    def __init__(self):
        self.borda_count_table = []
        self.test = False
        self.split_iterator_list = []

    def borda_count(self, parent, children, split_iterator):
        split_list = []
        split_list.append(Twoing(parent, children))
        split_list.append(QuinlanGain(parent, children))
        split_list.append(GiniImpurity(parent, children))
        split_list.append(MultiClassHellinger(parent, children))
        split_list.append(ChiSquared(parent, children))
        split_list.append(DKM(parent, children))
        split_list.append(G_Statistic(parent, children))
        split_list.append(MARSH(parent, children))
        split_list.append(NormalizedGain(parent, children))
        split_list.append(KolmogorovDependence(parent, children))
        self.borda_count_table.append(split_list)

    def borda_count_evaluate(self):
        if len(self.borda_count_table) == 0:
            return

        self.borda_count_table = np.array(self.borda_count_table)
        self.borda_count_table = np.transpose(self.borda_count_table)
        
        self.borda_count_table = pd.DataFrame.from_dict({
            'Twoing': self.borda_count_table[0],
            'Quinlan Gain': self.borda_count_table[1],
            'Gini Impurity': self.borda_count_table[2],
            'Multiclass Hellinger': self.borda_count_table[3],
            'Chi Squared': self.borda_count_table[4],
            'DKM': self.borda_count_table[5],
            'G Statistic': self.borda_count_table[6],
            'MARSH': self.borda_count_table[7],
            'Normalized Gain': self.borda_count_table[8],
            'Kolmogorov Dependence': self.borda_count_table[9]
        },
        orient='index', columns=[f'CS {i}' for i,_ in enumerate(self.borda_count_table[0])])

        print(f"{self.borda_count_table}\n")

        for index in self.borda_count_table.index:
            lst = self.borda_count_table.loc[index]
            sorted_indices = np.argsort(-lst)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(lst))
            self.borda_count_table.loc[index] = ranks

        #print(f"ranked:\n{self.borda_count_table}")
        rank_lst = [self.borda_count_table[f'{col}'].sum() for col in list(self.borda_count_table) ]
        #print(f"Rankings\n{rank_lst}")
        best_idx = smallest_idx(rank_lst)
        #print(f"smallest: {best_idx}")

        return smallest_idx(rank_lst)


