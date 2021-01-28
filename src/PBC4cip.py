import math
import numpy as np
from io import StringIO, BytesIO
from core.Helpers import ArgMax, convert_to_ndarray
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from PatternMiner import PatternMinerWithoutFiltering
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.DistributionTester import PureNodeStopCondition, AlwaysTrue
from core.Item import SubsetRelation
from core.Dataset import Dataset, FileDataset, PandasDataset
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate, obtainAUCMulticlass
from core.PatternFilter import MaximalPatternsGlobalFilter
from tqdm import tqdm


class PBC4cip:

    def __init__(self, tree_count=100, filtering=False, multivariate = False, file_dataset = None):
        self.File = None
        self.__miner = None
        if filtering:
            filterer = MaximalPatternsGlobalFilter()
            self.__filterer = filterer
        else:
            self.__filterer = None
        self.__filterer = None
        self.__multivariate = multivariate
        self.__treeCount = tree_count
        if file_dataset is not None:
            self.__dataset = FileDataset(file_dataset)
        else:
            self.__dataset = None            
        self.__EmergingPatterns = list()
        self.__class_nominal_feature = None
        self.__normalizing_vector = None
        self.__votesSum = None
        self.__classDistribution = None

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self.__dataset = new_dataset

    @property
    def multivariate(self):
        return self.__multivariate

    @multivariate.setter
    def multivariate(self, new_multivariate):
        self.__multivariate = new_multivariate
    
    @property
    def miner(self):
        return self.__miner

    @miner.setter
    def miner(self, new_miner):
        self.__miner = new_miner

    @property
    def filterer(self):
        return self.__filterer

    @filterer.setter
    def filterer(self, new_filterer):
        self.__filterer = new_filterer
        
    @property
    def treeCount(self):
        return self.__treeCount

    @treeCount.setter
    def treeCount(self, new_treeCount):
        self.__treeCount = new_treeCount


    def fit(self, X, y):
        if self.dataset is None:
            self.dataset = PandasDataset(X,y)
            X = X.to_numpy()
            y = y.to_numpy()
            if  not isinstance(y[0], np.ndarray):
                y = convert_to_ndarray(y)    

        self.miner = PatternMinerWithoutFiltering()
        miner = self.miner
        miner.dataset = self.dataset
        miner.TreeCount = self.treeCount
        if self.multivariate:
            miner.decisionTreeBuilder = MultivariateDecisionTreeBuilder(self.dataset, X, y)
            miner.decisionTreeBuilder.distributionEvaluator = QuinlanGain
        else:
            miner.decisionTreeBuilder = DecisionTreeBuilder(self.dataset, X, y)
            miner.decisionTreeBuilder.distributionEvaluator = QuinlanGain
        self.EmergingPatterns = miner.Mine()
        if self.filterer is not None:
            self.EmergingPatterns = self.filterer.Filter(self.EmergingPatterns)
        self.__ComputeVotes(X, y, self.dataset.Class[1])
        return self.EmergingPatterns

    def __predict_inst(self, instance):
        votes = [0]*len(self._class_nominal_feature)

        for pattern in self.EmergingPatterns:
            if pattern.IsMatch(instance):
                for i in range(len(votes)):
                    votes[i] += pattern.Supports[i]

        result = [0]*(len(votes)) 
        for i,_ in enumerate(votes):
            try:
                result[i] = votes[i] * \
                    self._normalizing_vector[i] / self.__votesSum[i]
            except ZeroDivisionError:
                result[i] = 0

        if sum(result) > 0:
            return result
        else:
            return self.__classDistribution
    
    def predict(self, X):
        if isinstance(self.dataset, PandasDataset):
            X = X.to_numpy()

        classification_results = list()
        for instance in tqdm(X, desc=f"Classifying instances", unit="instance", leave=False):
            result = self.__predict_inst(instance)
            classification_results.append(result)
        
        predicted = [ArgMax(instance) for instance in classification_results]
        return predicted
    
    def score(self, predicted, y):
        if isinstance(self.dataset, PandasDataset):
            y = y.to_numpy()
            y = convert_to_ndarray(y)
        real = list(map(lambda instance: self.dataset.GetClassValue(instance), y))
        numClasses = len(self._class_nominal_feature)
        confusion = [[0]*2 for i in range(numClasses)]
        classified_as = 0
        error_count = 0

        for i in range(len(real)):
            if real[i] != predicted[i]:
                error_count = error_count + 1
            confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

        acc = 100.0 * (len(real) - error_count) / len(real)
        auc = obtainAUCMulticlass(confusion, numClasses)

        return confusion, acc, auc

    
    def __ComputeVotes(self, X, y, classes):
        self._class_nominal_feature = classes
        instancesByClass = self.__GroupInstancesByClass(
            X, y, classes)
        self._normalizing_vector = self.__ComputeNormalizingVector(
            instancesByClass, len(y))
        self.__classDistribution = self.__ComputeClassDistribution(
            instancesByClass,  len(y))

        self.__votesSum = [0]*len(classes)
        for pattern in self.EmergingPatterns:
            for i,_ in enumerate(classes):
                self.__votesSum[i] += pattern.Supports[i]

    def __GroupInstancesByClass(self, X, y, classes):
        instancesByClass = list()

        for _ in enumerate(classes):
            instancesByClass.append(list())

        for i,instance in enumerate(y):
            instancesByClass[self.dataset.GetClassValue(instance)].append(X[i])

        return instancesByClass

    def __ComputeNormalizingVector(self, instancesByClass, instanceCount):
        vectorSum = 0
        normalizingVector = [0]*len(instancesByClass)

        for i, _ in enumerate(instancesByClass):
            try:
                normalizingVector[i] = 1.0 - 1.0 * \
                    len(instancesByClass[i]) / \
                    instanceCount
            except ZeroDivisionError:
                normalizingVector[i] = 0
            vectorSum += normalizingVector[i]

        for i, _ in enumerate(normalizingVector):
            try:
                normalizingVector[i] /= vectorSum
            except ZeroDivisionError:
                normalizingVector[i] = 0

        return normalizingVector

    def __ComputeClassDistribution(self, instancesByClass, instanceCount):
        classDistribution = [0]*len(instancesByClass)
        for i, _ in enumerate(instancesByClass):
            try:
                classDistribution[i] = 1.0 * \
                    len(instancesByClass[i]) / \
                    instanceCount
            except ZeroDivisionError:
                classDistribution[i] = 0

        return classDistribution


