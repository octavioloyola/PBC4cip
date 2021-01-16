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
from core.PatternFilter import MaximalPatternsGlobalFilter
from tqdm import tqdm


class PBC4cip:

    def __init__(self):
        #self.File = file
        self.File = None
        self.miner = None
        #self.Dataset = Dataset(self.File)
        #self.Dataset = None
        self.EmergingPatterns = list()
        self.TrainingSample = None
        self.MinimalPatternSupport = None
        self.ClassNominalFeature = None
        self.NormalizingVector = None
        self.__votesSum = None
        self.__classDistribution = None
    
    
    def set_dataset(self, file):
        self.Dataset = Dataset(file)
    
    def get_dataset(self):
        return self.Dataset
    
    def del_dataset(self):
        del self.Dataset

    def arg_max (self, source):
        idx = 0
        val = source[0]

        for (i, elem) in enumerate(source, start=0):
            if source[i] > val:
                idx = i
                val = source[i]
        
        return idx

    def Training(self, multivariate, filtering, trainFile, treeCount=None):
        #Sorta like Train inside PBC4cip
        #filtering = False
        dataset = Dataset(trainFile)
        print(f"Filtering: {filtering}")
        print(f"train: {dataset.Instances}")
        print(f"class: {dataset.Class}")
        miner = PatternMinerWithoutFiltering(dataset)
        """
        if filtering:
            miner = PatternMiner(self.Dataset)
        else:
            miner = PatternMinerWithoutFiltering(self.Dataset)
        """ 

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
        print(f"Patterns Have been found PBC4cip, ogLen: {len(self.EmergingPatterns)}")
        filterer = MaximalPatternsGlobalFilter()
        if filtering:
            #print(f"About to Filter Patterns")
            #print(f"OldPatterns:")
            #for pattern in self.EmergingPatterns:
                #print(f"{pattern}")

            #newPatterns = filterer.Filter_test(self.EmergingPatterns)
            self.EmergingPatterns = filterer.Filter(self.EmergingPatterns)
            
            #print(f"\nNewPatterns:")
            #for pattern in self.EmergingPatterns:
               #print(f"{pattern}")
        #print(f"Patterns that emerged: len: {len(self.EmergingPatterns)}\n{self.EmergingPatterns}\ntype: {type(self.EmergingPatterns)}")
        #print(f"num Patterns: {len(self.EmergingPatterns)}")
        #print(f"pattern 0: {self.EmergingPatterns[0]}")

        #dataset = self.EmergingPatterns[0].Dataset
        #self.TrainingSample = dataset.Instances
        #self.MinimalPatternSupport = 0

        #self.ComputeVotes()

        self.ComputeVotes(dataset)
        return self.EmergingPatterns

    def Classification(self, patterns, testInstances):
        #print(f"\nPatternsClassify: {patterns}")
        #print(f"typePatterns: {type(patterns)} typeof: {type(patterns[0])}")
        if not patterns or len(patterns) == 0:
            raise Exception(
                "In order to classify, previously extracted patterns are required.")

        train_dataset = patterns[0].Dataset
        test_dataset = Dataset(testInstances)

        print(f"LenTraining: {len(train_dataset.Instances)}") #Training
        print(f"LenTesting: {len(test_dataset.Instances)}") #Testing
        #print(f"Ex: {self.Dataset.Instances[0]} \n type: {type(self.Dataset.Instances[0])}, len: {len(self.Dataset.Instances[0])}")

        if not train_dataset:
            raise Exception(
                "In order to classify, training instances are required.")

        if test_dataset.Relation != train_dataset.Relation:
            raise Exception(
                "Patterns are not compatible with current testing dataset.")

        #self.EmergingPatterns = patterns
        #self.TrainingSample = train_dataset.Instances
        self.MinimalPatternSupport = 0
        
        #My changes here
        numPatterns = len(self.EmergingPatterns)
        print(f"numPatterns: {numPatterns}")
        patternLength = 0.0
        combinationLength = 0.0
        numClasses = len(test_dataset.Class)
        print(f"numClasses: {numClasses}")

        #self.ComputeVotes()  Original Place

        classification_results = list()

        for instance in tqdm(test_dataset.Instances, desc=f"Classifying instances for relation {test_dataset.Relation}", unit="instance", leave=False):
            #print("ffsdsd")
            result = self.Classify(instance)
            classification_results.append(result)

        #classified_as = 0
        #for instance in self.Dataset.Instances:
            #classification_result = self.Classify(instance)
            #for i, feature_val in enumerate(classification_result):
                #if classification_result[i] > classification_result[classified_as]:
                    #classified_as = i
            
            #confusion = []

        real = list(map(lambda instance: test_dataset.GetFeatureValue(test_dataset.Class, instance), test_dataset.Instances))
        predicted = [self.arg_max(instance) for instance in classification_results]

        #confusion, acc, auc = Evaluate(self.Dataset.Class[1], real, predicted)
        #print(f"Real: {real}, PredictionClass {self.Dataset.Class[1]}, real eval: {evaluation} ")
        
        #for prediction in predicted:
            #print(f"prediction: {prediction}")

        return Evaluate(test_dataset.Class[1], real, predicted)

    def Classify(self, instance):
        #print(f"NewInstance")
        #print(f"Enter Classify PBC4cip")
        votes = [0]*len(self.ClassNominalFeature[1])
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
            
    
    def ComputeVotes(self, training_dataset):
        #Compute Votes inside PBC4cip
        self.ClassNominalFeature = training_dataset.Class
        print(f"ClassFeature: {training_dataset.Class}")
        instancesByClass = self.GroupInstancesByClass(
            training_dataset)
        self.NormalizingVector = self.ComputeNormalizingVector(
            instancesByClass, len(training_dataset.Instances))
        self.__classDistribution = self.ComputeClassDistribution(
            instancesByClass,  len(training_dataset.Instances))

        self.__votesSum = [0]*len(training_dataset.Class[1])
        for pattern in self.EmergingPatterns:
            for classValue in range(len(training_dataset.Class[1])):
                self.__votesSum[classValue] += pattern.Supports[classValue]

    def GroupInstancesByClass(self, dataset):
        #print(f"Enter GroupInstancesByClass")
        instancesByClass = list()

        for classValue in range(len(dataset.Class[1])):
            instancesByClass.append(list())

        for instance in dataset.Instances:
            instancesByClass[dataset.Class[1].index(
                instance[dataset.GetClassIdx()])].append(instance)

        return instancesByClass

    def ComputeNormalizingVector(self, instancesByClass, instanceCount):
        #print(f"Enter ComputeNormalizing")
        vectorSum = 0
        normalizingVector = [0]*len(instancesByClass)

        for classValue in range(len(instancesByClass)):
            try:
                normalizingVector[classValue] = 1.0 - 1.0 * \
                    len(instancesByClass[classValue]) / \
                    instanceCount
            except ZeroDivisionError:
                normalizingVector[classValue] = 0
            vectorSum += normalizingVector[classValue]

        for index in range(len(normalizingVector)):
            try:
                normalizingVector[index] /= vectorSum
            except ZeroDivisionError:
                normalizingVector[index] = 0

        return normalizingVector

    def ComputeClassDistribution(self, instancesByClass, instanceCount):
        #print(f"Enter EnterComputeClass")
        classDistribution = [0]*len(instancesByClass)
        for classValue in range(len(instancesByClass)):
            try:
                classDistribution[classValue] = 1.0 * \
                    len(instancesByClass[classValue]) / \
                    instanceCount
            except ZeroDivisionError:
                classDistribution[classValue] = 0

        return classDistribution

