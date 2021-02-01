from .Item import SubsetRelation
from .EmergingPatterns import EmergingPatternComparer
from .Item import ItemComparer
from tqdm import tqdm

class MaximalPatternsGlobalFilter(object):
    def __init__(self):
       self.__comparer = EmergingPatternComparer(ItemComparer)

    @property
    def comparer(self):
        return self.__comparer
    @comparer.setter
    def comparer(self, new_comparer):
        self.__comparer = new_comparer

    def Filter(self, patterns):
        selected_patterns = set()

        for candidate_pattern in tqdm(patterns, desc=f"Filtering the found patterns", unit="candidate_pattern", leave=False ):
            minimal_patterns = set()
            general_pattern_found = False

            for selected_pattern in selected_patterns:
                pattern_relation = self.comparer.Compare(candidate_pattern, selected_pattern)
                if pattern_relation == SubsetRelation.Subset or pattern_relation == SubsetRelation.Equal:
                    general_pattern_found = True
                    break
                elif pattern_relation == SubsetRelation.Superset:
                    minimal_patterns.add(selected_pattern)
            
            if not general_pattern_found:
                for minimal_pattern in minimal_patterns:
                    selected_patterns.remove(minimal_pattern)
                selected_patterns.add(candidate_pattern)
        
        return list(selected_patterns)