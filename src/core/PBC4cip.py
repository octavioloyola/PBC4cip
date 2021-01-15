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
        filtering = False
        if filtering:
            miner = PatternMiner(self.Dataset)
        else:
            miner = PatternMinerWithoutFiltering(self.Dataset) 

        if not multivariate:
            print("nnot multivariate")
            miner.DecisionTreeBuilder = DecisionTreeBuilder(miner.Dataset)
            #miner.DecisionTreeBuilder.DistributionEvaluator = MultiClassHellinger
            miner.DecisionTreeBuilder.DistributionEvaluator = QuinlanGain
        else:
            print("multivariate")
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
        return self.EmergingPatterns

    def Classification(self, patterns):
        if not patterns or len(patterns) == 0:
            raise Exception(
                "In order to classify, previously extracted patterns are required.")

        dataset = patterns[0].Dataset

        if not dataset:
            raise Exception(
                "In order to classify, training instances are required.")

        if self.Dataset.Relation != dataset.Relation:
            raise Exception(
                "Patterns are not compatible with current testing dataset.")

        self.EmergingPatterns = patterns
        self.TrainingSample = dataset.Instances
        self.MinimalPatternSupport = 0

        self.Train()

        predicted = list()

        for instance in tqdm(self.Dataset.Instances, desc=f"Classifying instances for relation {self.Dataset.Relation}", unit="instance", leave=False):
            result = self.Classify(instance)
            predicted.append(result)

        real = list(map(lambda instance: self.Dataset.GetFeatureValue(
            self.Dataset.Class, instance), self.Dataset.Instances))

        evaluation = Evaluate(self.Dataset.Class[1], real, predicted)

        return evaluation

    def Classify(self, instance):
        classFeature = self.Dataset.Class
        votes = [0]*len(classFeature[1])

        for pattern in self.EmergingPatterns:
            if pattern.IsMatch(instance):
                for i in range(len(votes)):
                    votes[i] += pattern.Supports[i]

        result = [0]*(len(votes))
        for i in range(len(votes)):
            try:
                result[i] = votes[i] * \
                    self.NormalizingVector[i] / self.__votesSum[i]
            except ZeroDivisionError:
                result[i] = 0

        if sum(result) > 0:
            return result
        else:
            return self.__classDistribution

    def Train(self):
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
        instancesByClass = list()
        classIdx = self.Dataset.Model.index(classFeature)

        for classValue in range(len(classFeature[1])):
            instancesByClass.append(list())

        for instance in instances:
            instancesByClass[self.Dataset.Class[1].index(
                instance[classIdx])].append(instance)

        return instancesByClass

    def ComputeNormalizingVector(self, instancesByClass):
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
        classDistribution = [0]*len(instancesByClass)
        for classValue in range(len(instancesByClass)):
            try:
                classDistribution[classValue] = 1.0 * \
                    len(instancesByClass[classValue]) / \
                    len(self.TrainingSample)
            except ZeroDivisionError:
                classDistribution[classValue] = 0

        return classDistribution
