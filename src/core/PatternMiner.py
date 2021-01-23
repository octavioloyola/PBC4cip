import random
import math
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial
from core.SupervisedClassifier import DecisionTreeClassifier
from core.RandomSampler import SampleWithoutRepetition
from core.EmergingPatterns import EmergingPatternCreator, EmergingPatternComparer, EmergingPatternSimplifier
from core.DistributionTester import AlwaysTrue
from core.Item import ItemComparer, SubsetRelation
from core.FilteredCollection import FilteredCollection


class PatternMinerWithoutFiltering:

    def __init__(self, treeCount=None, featureCount=None, minePatternsWhileBuildingTree=None):
        self._dataset = None
        self.Patterns = list()
        self.PatternsList = list()
        self._decisionTreeBuilder = None
        self.EPTester = AlwaysTrue
        self.FilterRelation = SubsetRelation.Superset

        self.__emergingPatternCreator = None
        self.__emergingPatternComparer = None
        self.__emergingPatternSimplifier = None
        self.__minimal = None

        if not featureCount:
            self.FeatureCount = -1
        else:
            self.FeatureCount = featureCount

        if not treeCount:
            self.TreeCount = 100
        else:
            self.TreeCount = treeCount

        if not minePatternsWhileBuildingTree:
            self.MinePatternsWhileBuildingTree = False
        else:
            self.MinePatternsWhileBuildingTree = minePatternsWhileBuildingTree

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset
    
    @property
    def decisionTreeBuilder(self):
        return self._decisionTreeBuilder
    
    @decisionTreeBuilder.setter
    def decisionTreeBuilder(self, new_dtb):
        self._decisionTreeBuilder = new_dtb

    def Mine(self):
        self.Patterns = list()
        self.__emergingPatternCreator = EmergingPatternCreator(self.dataset)
        self.__emergingPatternComparer = EmergingPatternComparer(
            ItemComparer().Compare)
        self.__emergingPatternSimplifier = EmergingPatternSimplifier(
            ItemComparer().Compare)
        self.Patterns = self.__DoMine(
            self.__emergingPatternCreator, self.PatternFound)
        self.PatternsList = []

        return self.Patterns

    def __DoMine(self, emergingPatternCreator, action):
        freeze_support()  # for Windows support
        featureCount = 0
        if self.FeatureCount != -1:
            featureCount = self.FeatureCount
        else:
            featureCount = int(math.log(len(self.dataset.Attributes), 2) + 1)

        decision_tree_builder = self._decisionTreeBuilder
        decision_tree_builder.FeatureCount = featureCount
        decision_tree_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
        for i in tqdm(range(self.TreeCount), unit="tree", desc="Building trees and extracting patterns", leave=False):
            decision_tree_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
            tree = decision_tree_builder.Build()
            treeClassifier = DecisionTreeClassifier(tree)
            emergingPatternCreator.ExtractPatterns(treeClassifier, action)

        return self.PatternsList

    def PatternFound(self, pattern):        
        if self.EPTester(pattern.Counts, self._dataset.Model, self._dataset.Class):
            simplifiedPattern = self.__emergingPatternSimplifier.Simplify(pattern)
            self.PatternsList.append(simplifiedPattern)
                