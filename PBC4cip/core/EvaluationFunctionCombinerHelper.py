from .DistributionEvaluator import Hellinger, MultiClassBhattacharyya
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence
def get_functions_dict(func_names):
        func_names = [name.lower() for name in func_names]
        evaluator_dict = {
                'twoing': Twoing,
                'quinlan gain': QuinlanGain,
                'gini impurity': GiniImpurity,
                'multi class hellinger': MultiClassHellinger,
                'chi squared': ChiSquared,
                'dkm': DKM,
                'g statistic': G_Statistic,
                'marsh': MARSH,
                'normalized gain': NormalizedGain,
                'kolmogorov': KolmogorovDependence,
                'bhattacharyya': MultiClassBhattacharyya
            }
        return {key:value for (key,value) in evaluator_dict.items() if key in func_names}
