from core.Item import SubsetRelation
from core.EmergingPatterns import EmergingPatternComparer
from core.Item import ItemComparer
from tqdm import tqdm

class MaximalPatternsGlobalFilter(object):
    def __init__(self):
       self.comparer = EmergingPatternComparer(ItemComparer)
       self.counter = 0 


    def Filter(self, patterns):
        selected_patterns = set()

        for candidate_pattern in tqdm(patterns, desc=f"Filtering the found patterns", unit="candidate_pattern", leave=False ):
            minimal_patterns = set()
            general_pattern_found = False
            self.counter = self.counter +1

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