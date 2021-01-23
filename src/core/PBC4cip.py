import math
from io import StringIO, BytesIO
from core.Helpers import ArgMax
from core.DecisionTreeBuilder import DecisionTreeBuilder, MultivariateDecisionTreeBuilder
from core.PatternMiner import PatternMinerWithoutFiltering
from core.DistributionEvaluator import Hellinger, MultiClassHellinger, QuinlanGain
from core.DistributionTester import PureNodeStopCondition, AlwaysTrue
from core.Item import SubsetRelation
from core.Dataset import Dataset
from core.Evaluation import CrispAndPartitionEvaluation, Evaluate
from core.PatternFilter import MaximalPatternsGlobalFilter
from tqdm import tqdm


class PBC4cip:

    def __init__(self, treeCount=None):
        self.File = None
        self._miner = None
        self._filterer = None
        self._multivariate = None
        self._treeCount = treeCount
        self._dataset = None
        self.EmergingPatterns = list()
        self.TrainingSample = None
        self.ClassNominalFeature = None
        self.NormalizingVector = None
        self.__votesSum = None
        self.__classDistribution = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset

    @property
    def multivariate(self):
        return self._multivariate

    @multivariate.setter
    def multivariate(self, new_multivariate):
        self._multivariate = new_multivariate
    
    @property
    def miner(self):
        return self._miner

    @miner.setter
    def miner(self, new_miner):
        self._miner = new_miner

    @property
    def filterer(self):
        return self._filterer

    @filterer.setter
    def filterer(self, new_filterer):
        self._filterer = new_filterer


    def fit(self, X, y):
        miner = self.miner
        miner.TreeCount = self._treeCount

        self.EmergingPatterns = miner.Mine()
        if self.filterer is not None:
            self.EmergingPatterns = filterer.Filter(self.EmergingPatterns)
        self.__ComputeVotes(X, y, self._dataset.Class[1])
        return self.EmergingPatterns

    def predict(self, instance):
        votes = [0]*len(self.ClassNominalFeature)

        for pattern in self.EmergingPatterns:
            if pattern.IsMatch(instance):
                for i in range(len(votes)):
                    votes[i] += pattern.Supports[i]

        result = [0]*(len(votes)) 
        for i,_ in enumerate(votes):
            try:
                result[i] = votes[i] * \
                    self.NormalizingVector[i] / self.__votesSum[i]
            except ZeroDivisionError:
                result[i] = 0

        if sum(result) > 0:
            return result
        else:
            return self.__classDistribution
    
    def __ComputeVotes(self, X, y, classes):
        self.ClassNominalFeature = classes
        instancesByClass = self.__GroupInstancesByClass(
            X, y, classes)
        self.NormalizingVector = self.__ComputeNormalizingVector(
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

        for instance in self._dataset.Instances:
            instancesByClass[self._dataset.Class[1].index(
                instance[self._dataset.GetClassIdx()])].append(instance)

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

