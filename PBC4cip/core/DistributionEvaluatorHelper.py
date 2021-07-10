from .DistributionEvaluator import Hellinger, MultiClassBhattacharyya
from .DistributionEvaluator import Twoing, QuinlanGain, GiniImpurity, MultiClassHellinger, ChiSquared
from .DistributionEvaluator import DKM, G_Statistic, MARSH, NormalizedGain, KolmogorovDependence
from .EvaluationFunctionCombiner import EvaluationFunctionCombiner
from .EvaluationFunctionCombinerRandom import EvaluationFunctionCombinerRandom
from .InstantRunoffVoting import InstantRunoffVoting
def get_distribution_evaluator(eval_func_name):
        evaluator_dict = {
            'twoing': Twoing,
            'quinlan gain': QuinlanGain,
            'qg': QuinlanGain,
            'gini impurity': GiniImpurity,
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
            'combiner random': EvaluationFunctionCombinerRandom,
            'irv': InstantRunoffVoting
        }

        if eval_func_name.lower().replace('-', ' ') in evaluator_dict:
            return evaluator_dict[eval_func_name.lower().replace('-', ' ')]
        else:
            raise Exception(f"{eval_func_name.lower()} is not a supported evaluation function")
    
