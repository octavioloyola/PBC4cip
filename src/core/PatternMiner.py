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
        self.__dataset = None
        self.__Patterns = list()
        self.__PatternsList = list()
        self.__decisionTreeBuilder = None
        self.__EPTester = AlwaysTrue
        self.__FilterRelation = SubsetRelation.Superset

        self.__emergingPatternCreator = None
        self.__emergingPatternComparer = None
        self.__emergingPatternSimplifier = None
        self.__minimal = None

        if not featureCount:
            self.__FeatureCount = -1
        else:
            self.__FeatureCount = featureCount

        if not treeCount:
            self.__TreeCount = 100
        else:
            self.__TreeCount = treeCount

        if not minePatternsWhileBuildingTree:
            self.__MinePatternsWhileBuildingTree = False
        else:
            self.__MinePatternsWhileBuildingTree = minePatternsWhileBuildingTree

    @property
    def TreeCount(self):
        return self.__TreeCount
    @TreeCount.setter
    def TreeCount(self, new_tree_count):
        self.__TreeCount = new_tree_count

    @property
    def dataset(self):
        return self.__dataset
    @dataset.setter
    def dataset(self, new_dataset):
        self.__dataset = new_dataset
    
    @property
    def decisionTreeBuilder(self):
        return self.__decisionTreeBuilder
    @decisionTreeBuilder.setter
    def decisionTreeBuilder(self, new_dtb):
        self.__decisionTreeBuilder = new_dtb

    @property
    def FeatureCount(self):
        return self.__FeatureCount
    @FeatureCount.setter
    def FeatureCount(self, new_Feature_Count):
        self.__FeatureCount = new_Feature_Count
    
    @property
    def EPTester(self):
        return self.__EPTester

    @property
    def PatternsList(self):
        return self.__PatternsList

    @PatternsList.setter
    def PatternsList(self, new_patterns_list):
        self.__PatternsList = new_patterns_list

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

        decision_tree_builder = self.decisionTreeBuilder
        decision_tree_builder.FeatureCount = featureCount
        decision_tree_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
        for i in tqdm(range(self.TreeCount), unit="tree", desc="Building trees and extracting patterns", leave=False):
            decision_tree_builder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
            tree = decision_tree_builder.Build()
            treeClassifier = DecisionTreeClassifier(tree)
            emergingPatternCreator.ExtractPatterns(treeClassifier, action)

        return self.PatternsList

    def PatternFound(self, pattern):        
        if self.EPTester(pattern.Counts, self.dataset.Model, self.dataset.Class):
            simplifiedPattern = self.__emergingPatternSimplifier.Simplify(pattern)
            self.PatternsList.append(simplifiedPattern)
                