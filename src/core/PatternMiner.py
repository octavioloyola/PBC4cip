import random
import math
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial
from core.SupervisedClassifier import DecisionTreeClassifier
from core.RandomSampler import SampleWithoutRepetition
from core.EmergingPatterns import EmergingPatternCreator, EmergingPatternComparer, EmergingPatternSimplifier
from core.Item import ItemComparer
from core.FilteredCollection import FilteredCollection


class PatternMiner:

    def __init__(self, dataset, treeCount=None, featureCount=None, minePatternsWhileBuildingTree=None):
        print("Mining with Filter")
        self.Dataset = dataset
        self.Patterns = list()
        self.DecisionTreeBuilder = None
        self.EPTester = None
        self.FilterRelation = None

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

    # region Pattern extraction

    def Mine(self):
        print("dsdsdsdsaszxzxdd")
        #print(self.PatternFound)
        self.Patterns = list()
        self.__emergingPatternCreator = EmergingPatternCreator(self.Dataset)
        self.__emergingPatternComparer = EmergingPatternComparer(
            ItemComparer().Compare)
        self.__emergingPatternSimplifier = EmergingPatternSimplifier(
            ItemComparer().Compare)
        self.__minimal = FilteredCollection(
            self.__emergingPatternComparer.Compare, self.FilterRelation)
        self.Patterns = self.DoMine(
            self.__emergingPatternCreator, self.PatternFound)
        return self.Patterns

    def DoMine(self, emergingPatternCreator, action):
        print("does it enter DoMine")
        freeze_support()  # for Windows support
        featureCount = 0
        if self.FeatureCount != -1:
            featureCount = self.FeatureCount
        else:
            featureCount = int(math.log(len(self.Dataset.Attributes), 2) + 1)

        self.DecisionTreeBuilder.FeatureCount = featureCount
        print(f"amount of featuresToConsider: {self.Dataset.Attributes}")
        self.DecisionTreeBuilder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
        print(f"featureCount: {self.DecisionTreeBuilder.FeatureCount} and {featureCount}")

        #print(f"amount of leaves: {type(tree.Leaves())}")

        print(f"amount of Trees: {self.TreeCount}")
        for i in tqdm(range(1), unit="tree", desc="Building trees and extracting patterns", leave=False):
            self.DecisionTreeBuilder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
            tree = self.DecisionTreeBuilder.Build()
            #print(f"amount of leaves: {type(tree.Leaves)}")
            treeClassifier = DecisionTreeClassifier(tree)
            #up until here everything is ok!!
            emergingPatternCreator.ExtractPatterns(treeClassifier, action)

        #print(f"typeOfMinimal: {type(self.__minimal)}")
        #print(f"minimal.getItems: {self.__minimal.GetItems()}")
        return self.__minimal.GetItems()

    # endregion

    def CreateTreeAndExtractpatterns(self, emergingPatternCreator, action, iterable):
        print("Create Tree and Extract Patterns")
        tree = self.DecisionTreeBuilder.Build()
        treeClassifier = DecisionTreeClassifier(tree)
        emergingPatternCreator.ExtractPatterns(treeClassifier, action)

    def PatternFound(self, pattern):
        print(f"pattern Found: {pattern}")
        #self.__minimal.Add(pattern)
        
        if self.EPTester(pattern.Counts, self.Dataset.Model, self.Dataset.Class):
            #print(f"sssssss")
            #simplifiedPattern = self.__emergingPatternSimplifier.Simplify(pattern)
            #print(f"simplified Pattern: {simplifiedPattern}")
            self.__minimal.Add(
            #pattern)
            self.__emergingPatternSimplifier.Simplify(pattern))
            #    simplifiedPattern)
                
        #print(f"size of minimal (in lambda): {len(self.__minimal.GetItems())}")
        #print(f"minimal (lambda): {self.__minimal.GetItems()}\n \n")

class PatternMinerWithoutFiltering:

    def __init__(self, dataset, treeCount=None, featureCount=None, minePatternsWhileBuildingTree=None):
        print("Pattern Miner without filter init ")
        self.Dataset = dataset
        self.Patterns = list()
        self.PatternsList = list()
        self.DecisionTreeBuilder = None
        self.EPTester = None
        self.FilterRelation = None

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

    # region Pattern extraction

    def Mine(self):
        #print("Without Filtering Mine begins")
        #print(self.PatternFound)
        #print(f"ItemComparer: {ItemComparer}")
        self.Patterns = list()
        self.__emergingPatternCreator = EmergingPatternCreator(self.Dataset)
        self.__emergingPatternComparer = EmergingPatternComparer(
            ItemComparer().Compare)
        self.__emergingPatternSimplifier = EmergingPatternSimplifier(
            ItemComparer().Compare)
        #self.__minimal = FilteredCollection(
            #self.__emergingPatternComparer.Compare, self.FilterRelation)
        self.Patterns = self.DoMine(
            self.__emergingPatternCreator, self.PatternFound)
        self.PatternsList = []
        #patternsList = []

        return self.Patterns

    def DoMine(self, emergingPatternCreator, action):
        #print("does it enter DoMine Without Filtering")
        #patternsList = []
        freeze_support()  # for Windows support
        featureCount = 0
        if self.FeatureCount != -1:
            featureCount = self.FeatureCount
        else:
            featureCount = int(math.log(len(self.Dataset.Attributes), 2) + 1)

        self.DecisionTreeBuilder.FeatureCount = featureCount
        #print(f"amount of featuresToConsider: {self.Dataset.Attributes}")
        #print(f"Dataset class: {self.Dataset.ClassInformation.Feature}")
        self.DecisionTreeBuilder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
        #print(f" featureCount: {self.DecisionTreeBuilder.FeatureCount} and {featureCount}")

        #print(f"amount of leaves: {type(tree.Leaves())}")

        #print(f"amount of Trees: {self.TreeCount}")

        ##self.DecisionTreeBuilder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
        ##tree = self.DecisionTreeBuilder.Build()
        ##treeClassifier = DecisionTreeClassifier(tree)

        for i in tqdm(range(self.TreeCount), unit="tree", desc="Building trees and extracting patterns", leave=False):
        #for i in range(self.TreeCount):    
            #print(f"tree: {i}")
            self.DecisionTreeBuilder.OnSelectingFeaturesToConsider = SampleWithoutRepetition
            #print(f"featuresToConsider: {self.DecisionTreeBuilder.OnSelectingFeaturesToConsider}")
            tree = self.DecisionTreeBuilder.Build()
            #classFeature = self.Dataset.ClassInformation.Feature
            #print(f"amount of leaves: {type(tree.Leaves)}")
            treeClassifier = DecisionTreeClassifier(tree)
            #up until here everything is ok!!
            #print(f"about to extract Patterns in tree")
            emergingPatternCreator.ExtractPatterns(treeClassifier, action)

        #print(f"PatternsListLen: {len(self.PatternsList)}")
        #print(f"PatternsList[0]: {self.PatternsList[0]}")
        return self.PatternsList

    # endregion

    def CreateTreeAndExtractpatterns(self, emergingPatternCreator, action, iterable):
        #print("Create Tree and Extract Patterns")
        tree = self.DecisionTreeBuilder.Build()
        treeClassifier = DecisionTreeClassifier(tree)
        emergingPatternCreator.ExtractPatterns(treeClassifier, action)

    def PatternFound(self, pattern):
        #print(f"\npattern Found no filter: {pattern}")
        #self.__minimal.Add(pattern)
        
        if self.EPTester(pattern.Counts, self.Dataset.Model, self.Dataset.Class):
            #print(f"sssssss")
            simplifiedPattern = self.__emergingPatternSimplifier.Simplify(pattern)
            #print(f"simplified Pattern: {simplifiedPattern}")
            self.PatternsList.append(simplifiedPattern)
                
        #print(f"size of patternsList (in lambda): {len(self.PatternsList)}")
        #print(f"patternsList Contents: {self.PatternsList}\n \n")
