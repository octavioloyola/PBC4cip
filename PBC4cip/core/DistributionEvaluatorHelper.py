from .DistributionEvaluator import Hellinger, MultiClassBhattacharyya
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence
from .EvaluationFunctionCombiner import EvaluationFunctionCombiner
from .EvaluationFunctionCombinerRandom import EvaluationFunctionCombinerRandom
def get_distribution_evaluator(eval_func_name):
        evaluator_dict = {
            'twoing': Twoing,
            'quinlan': QuinlanGain,
            'gini': GiniImpurity,
            'hellinger': Hellinger,
            'multi class hellinger': MultiClassHellinger,
            'chi squared': ChiSquared,
            'dkm': DKM,
            'g statistic': G_Statistic,
            'marsh': MARSH,
            'normalized gain': NormalizedGain,
            'kolmogorov': KolmogorovDependence,
            'bhattacharyya': MultiClassBhattacharyya,
            'combiner': EvaluationFunctionCombiner,
            'combiner-random': EvaluationFunctionCombinerRandom
        }

        if eval_func_name.lower() in eval_func_name:
            return evaluator_dict[eval_func_name.lower()]
        else:
            raise Exception(f"{eval_func_name} is not a supported evaluation function")
    
