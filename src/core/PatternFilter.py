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
        print("Begin MaximalPatternsGlobalFilter")

        for candidate_pattern in tqdm(patterns, desc=f"Filtering the found patterns", unit="candidate_pattern", leave=False ):
            #print(f"typePattern: {type(candidate_pattern)}")
            #print(f"Candidate: {candidate_pattern}")
            minimal_patterns = set()
            general_pattern_found = False
            self.counter = self.counter +1
            #print(f"counter: {self.counter}" , end=", ")

            for selected_pattern in selected_patterns:
                pattern_relation = self.comparer.Compare(candidate_pattern, selected_pattern)
                if pattern_relation == SubsetRelation.Subset or pattern_relation == SubsetRelation.Equal:
                    #print(f"general_pattern_found: Candidate: {candidate_pattern} selected: {selected_pattern}")
                    #if pattern_relation == SubsetRelation.Subset: print("Subset")
                    #if pattern_relation == SubsetRelation.Equal: print("Equal")

                    #pattern_relation_test = self.comparer.Compare_test(candidate_pattern, selected_pattern)
                    general_pattern_found = True
                    break
                elif pattern_relation == SubsetRelation.Superset:
                    minimal_patterns.add(selected_pattern)
            
            if not general_pattern_found:
                for minimal_pattern in minimal_patterns:
                    selected_patterns.remove(minimal_pattern)
                selected_patterns.add(candidate_pattern)
        
        return list(selected_patterns)

    def Filter_test(self, patterns):
        #patterns = EmergingPatternCreator.
        selected_patterns = set()
        print("")
        for candidate_pattern in patterns:
            
            #print(type(candidate_pattern))
            #print(f"Candidate: {candidate_pattern}")
            minimal_patterns = set()
            general_pattern_found = False

            for selected_pattern in selected_patterns:
                self.counter = self.counter+1
                pattern_relation = self.comparer.Compare(candidate_pattern, selected_pattern)
                pattern_relation_test = self.comparer.Compare_test(candidate_pattern, selected_pattern)
                #print(f"counter {self.counter}")
                #if self.counter == 5:
                    #print(f"Test: selected:{selected_pattern}")
                    #pattern_relation_test = self.comparer.Compare_test(selected_pattern, selected_pattern)
                if pattern_relation == SubsetRelation.Subset or pattern_relation == SubsetRelation.Equal:
                    #print(f"general_pattern_found: Candidate: {candidate_pattern} selected: {selected_pattern}")
                    #if pattern_relation == SubsetRelation.Subset: print("Subset")
                    #if pattern_relation == SubsetRelation.Equal: print("Equal")

                    
                    general_pattern_found = True
                    break
                elif pattern_relation == SubsetRelation.Superset:
                    minimal_patterns.add(selected_pattern)
            
            if not general_pattern_found:
                for minimal_pattern in minimal_patterns:
                    selected_patterns.remove(minimal_pattern)
                selected_patterns.add(candidate_pattern)
        
        return list(selected_patterns)


