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
        #print(f"selff {self.borda_count_table}")

    def borda_count(self, parent, children, split_iterator):
        if self.test:
            parent_test = [129.0, 10.0]
            children_test = [[110.0, 6.0], [19.0, 4.0]]
            print(f"Twoing: {Twoing(parent_test, children_test)}")
            print(f"Quinlan Gain: {QuinlanGain(parent_test, children_test)}")
            print(f"Gini Impurity: {GiniImpurity(parent_test, children_test)}")
            print(f"Multiclass Hellinger: {MultiClassHellinger(parent_test, children_test)}")
            print(f"Chi Squared: {ChiSquared(parent_test, children_test)}")
            print(f"DKM: {DKM(parent_test, children_test)}")
            print(f"G_Statistic: {G_Statistic(parent_test, children_test)}")
            print(f"MARSH: {MARSH(parent_test, children_test)}")
            print(f"Normalized Gain: {NormalizedGain(parent_test, children_test)}")
            print(f"KolmogorovDependence: {KolmogorovDependence(parent_test, children_test)}")
            self.test = False
        #print(f"bordaCount {self.borda_count_table}")
        
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
            #print(self.borda_count_table.loc[index])
            lst = self.borda_count_table.loc[index]
            sorted_indices = np.argsort(-lst)
            #print(f"lst:\n{lst}")
            #print(f"sorted_indices:\n{sorted_indices}")
            ranks = np.empty_like(sorted_indices)
            #print(f"ranks:\n{ranks}")
            ranks[sorted_indices] = np.arange(len(lst))
            #print(f"ranks:\n{ranks}")
            self.borda_count_table.loc[index] = ranks
            #print(self.borda_count_table.loc[[index]])

        print(f"ranked:\n{self.borda_count_table}")
        rank_lst = [self.borda_count_table[f'{col}'].sum() for col in list(self.borda_count_table) ]
        print(f"Rankings\n{rank_lst}")
        best_idx = smallest_idx(rank_lst)
        print(f"smallest: {best_idx}")

        return smallest_idx(rank_lst)


