# -*- coding: utf-8 -*-
import math
from io import StringIO, BytesIO
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMiner, PatternMinerWithoutFiltering
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.DistributionTester import PureNodeStopCondition, AlwaysTrue
from core.Item import SubsetRelation
from core.Dataset import Dataset
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate
from tqdm import tqdm


class PBC4cip:

    def __init__(self, file):
        self.File = file
        self.Dataset = Dataset(self.File)
        self.EmergingPatterns = list()
        self.TrainingSample = None
        self.MinimalPatternSupport = None
        self.NormalizingVector = None
        self.__votesSum = None
        self.__classDistribution = None

    def Training(self, multivariate, treeCount=None):
        #Sorta like Train inside PBC4cip
        filtering = False
        if filtering:
            miner = PatternMiner(self.Dataset)
        else:
            miner = PatternMinerWithoutFiltering(self.Dataset) 

        if not multivariate:
            print("nnot multivariate has been set")
            miner.DecisionTreeBuilder = DecisionTreeBuilder(miner.Dataset)
            #miner.DecisionTreeBuilder.DistributionEvaluator = MultiClassHellinger
            #miner.DecisionTreeBuilder.DistributionEvaluator = Hellinger
            miner.DecisionTreeBuilder.DistributionEvaluator = QuinlanGain
        else:
            print("multivariate miner has been set")
            miner.DecisionTreeBuilder = MultivariateDecisionTreeBuilder(miner.Dataset)
            miner.DecisionTreeBuilder.DistributionEvaluator = MultiClassHellinger

        miner.DecisionTreeBuilder.StopCondition = PureNodeStopCondition

        if not treeCount:
            miner.TreeCount = 100
        else:
            miner.TreeCount = treeCount
        miner.EPTester = AlwaysTrue
        miner.MinePatternsWhileBuildingTree = False
        miner.FilterRelation = SubsetRelation.Superset

        self.EmergingPatterns = miner.Mine()
        #print(f"Patterns that emerged: len: {len(self.EmergingPatterns)}\n{self.EmergingPatterns}\ntype: {type(self.EmergingPatterns)}")
        print(f"num Patterns: {len(self.EmergingPatterns)}")
        print(f"pattern 0: {self.EmergingPatterns[0]}")

        #self.EmergingPatterns = patterns
        #self.TrainingSample = dataset.Instances
        #self.MinimalPatternSupport = 0

        #self.ComputeVotes()
        return self.EmergingPatterns

    def Classification(self, patterns):
        print(f"\nPatternsClassify: {patterns}")
        print(f"typePatterns: {type(patterns)} typeof: {type(patterns[0])}")
        if not patterns or len(patterns) == 0:
            raise Exception(
                "In order to classify, previously extracted patterns are required.")

        dataset = patterns[0].Dataset

        print(f"Dataset: {len(dataset.Instances)}") #Training
        print(f"DatasetSelf: {len(self.Dataset.Instances)}") #Testing
        print(f"Ex: {self.Dataset.Instances[0]} \n type: {type(self.Dataset.Instances[0])}, len: {len(self.Dataset.Instances[0])}")

        if not dataset:
            raise Exception(
                "In order to classify, training instances are required.")

        if self.Dataset.Relation != dataset.Relation:
            raise Exception(
                "Patterns are not compatible with current testing dataset.")

        self.EmergingPatterns = patterns
        self.TrainingSample = dataset.Instances
        self.MinimalPatternSupport = 0
        
        #My changes here
        numPatterns = len(self.EmergingPatterns)
        print(f"numPatterns: {numPatterns}")
        patternLength = 0.0
        combinationLength = 0.0
        numClasses = len(self.Dataset.Class)
        print(f"numClasses: {numClasses}")

        self.ComputeVotes()

        predicted = list()

        for instance in tqdm(self.Dataset.Instances, desc=f"Classifying instances for relation {self.Dataset.Relation}", unit="instance", leave=False):
            result = self.Classify(instance)
            predicted.append(result)

        #classified_as = 0
        #for instance in self.Dataset.Instances:
            #classification_result = self.Classify(instance)
            #for i, feature_val in enumerate(classification_result):
                #if classification_result[i] > classification_result[classified_as]:
                    #classified_as = i
            
            #confusion = []

        real = list(map(lambda instance: self.Dataset.GetFeatureValue(self.Dataset.Class, instance), self.Dataset.Instances))

        #confusion, acc, auc = Evaluate(self.Dataset.Class[1], real, predicted)
        #print(f"Real: {real}, PredictionClass {self.Dataset.Class[1]}, real eval: {evaluation} ")
        
        #for prediction in predicted:
            #print(f"prediction: {prediction}")

        return Evaluate(self.Dataset.Class[1], real, predicted)

    def Classify(self, instance):
        #print(f"NewInstance")
        #print(f"Enter Classify PBC4cip")
        classFeature = self.Dataset.Class
        votes = [0]*len(classFeature[1])
        #print(f"lenClassFeat: {len(classFeature[1])}")

        for pattern in self.EmergingPatterns:
            #if (instance[0] == 41.0):
            #print(f"\nep: {pattern}")
            #print(f"epItems: {pattern.Items}")
            if pattern.IsMatch(instance):
                #print(f"MatchFound!")
                for i in range(len(votes)):
                    votes[i] += pattern.Supports[i]
        
        #if (instance[0] == 41.0):
        #print(f"votes: {votes}")

        result = [0]*(len(votes))
        for i in range(len(votes)):
            try:
                result[i] = votes[i] * \
                    self.NormalizingVector[i] / self.__votesSum[i]
            except ZeroDivisionError:
                result[i] = 0

        #if (instance[0] == 41.0):
        #print(f"instance: {instance}")
        #print(f"result: {result}")

        if sum(result) > 0:
            return result
        else:
            return self.__classDistribution
            

    def ComputeVotes(self):
        #Compute Votes inside PBC4cip
        #print(f"Enter ComputeVotes")
        #print(f"trainingSample: {self.TrainingSample}")
        instancesByClass = self.GroupInstancesByClass(
            self.TrainingSample, self.Dataset.Class)
        self.NormalizingVector = self.ComputeNormalizingVector(
            instancesByClass)
        self.__classDistribution = self.ComputeClassDistribution(
            instancesByClass)

        self.__votesSum = [0]*len(self.Dataset.Class[1])
        for pattern in self.EmergingPatterns:
            for classValue in range(len(self.Dataset.Class[1])):
                self.__votesSum[classValue] += pattern.Supports[classValue]

    def GroupInstancesByClass(self, instances, classFeature):
        #print(f"Enter GroupInstancesByClass")
        instancesByClass = list()
        classIdx = self.Dataset.Model.index(classFeature)

        for classValue in range(len(classFeature[1])):
            instancesByClass.append(list())

        for instance in instances:
            instancesByClass[self.Dataset.Class[1].index(
                instance[classIdx])].append(instance)

        return instancesByClass

    def ComputeNormalizingVector(self, instancesByClass):
        #print(f"Enter ComputeNormalizing")
        vetorSum = 0
        normalizingVector = [0]*len(instancesByClass)

        for classValue in range(len(instancesByClass)):
            try:
                normalizingVector[classValue] = 1.0 - 1.0 * \
                    len(instancesByClass[classValue]) / \
                    len(self.TrainingSample)
            except ZeroDivisionError:
                normalizingVector[classValue] = 0
            vetorSum += normalizingVector[classValue]

        for index in range(len(normalizingVector)):
            try:
                normalizingVector[index] /= vetorSum
            except ZeroDivisionError:
                normalizingVector[index] = 0

        return normalizingVector

    def ComputeClassDistribution(self, instancesByClass):
        #print(f"Enter EnterComputeClass")
        classDistribution = [0]*len(instancesByClass)
        for classValue in range(len(instancesByClass)):
            try:
                classDistribution[classValue] = 1.0 * \
                    len(instancesByClass[classValue]) / \
                    len(self.TrainingSample)
            except ZeroDivisionError:
                classDistribution[classValue] = 0

        return classDistribution
