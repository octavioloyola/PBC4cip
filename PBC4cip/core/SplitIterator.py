def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import math
from .Helpers import FindDistribution, Substract
import operator
from .FeatureSelectors import CutPointSelector, MultipleValuesSelector, ValueAndComplementSelector, MultivariateCutPointSelector
from copy import copy, deepcopy
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class SplitIteratorProvider(object):
    def __init__(self, dataset):
        self.Dataset = dataset

    def GetSplitIterator(self, feature):
        currentFeature = self.Dataset.GetAttribute(feature)
        if self.Dataset.IsNominalFeature(currentFeature):
            return NominalSplitIterator(self.Dataset, currentFeature)
        else:
            return NumericSplitIterator(self.Dataset, currentFeature)


class SplitIterator(object):
    def __init__(self, dataset, feature):
        self.__Dataset = dataset
        self.__Model = self.Dataset.Model
        self.__Class = self.Dataset.Class
        self.__Feature = feature
        self.__CurrentDistribution = None
        self.__initialized = False
        self.__numClasses = 0
        self.__instances = 0

    @property
    def Dataset(self):
        return self.__Dataset
    @Dataset.setter
    def Dataset(self, new_dataset):
        self.__Dataset = new_dataset
    
    @property
    def Model(self):
        return self.__Model
    @Model.setter
    def Model(self, new_model):
        self.__Model = new_model
    
    @property
    def Class(self):
        return self.__Class
    @Class.setter
    def Class(self, new_class):
        self.__Class = new_class
    
    @property
    def Feature(self):
        return self.__Feature
    @Feature.setter
    def Feature(self, new_feature):
        self.__Feature = new_feature

    def Initialize(self, instances):
        if not self.Model:
            raise Exception("Model is null")
        if self.Class[1] in ['numeric', 'real', 'integer', 'string']:
            raise Exception("Cannot use this iterator on non-nominal class")
        self._numClasses = len(self.Dataset.GetClasses())
        self._instances = len(instances)
        self.CurrentDistribution = [[] for i in range(2)]
        self._initialized = True

    def FindNext(self):
        if not self._initialized:
            raise Exception("Iterator not initialized")
        return False

    def CreateCurrentChildSelector(self):
        return None
    def GetFeatureIdx(self):
        return self.Dataset.GetFeatureIdx(self.Feature)

    def IsMissing(self, instance):
        return self.Dataset.IsMissing(self.Feature, instance)

    def GetFeatureValue(self, instance):
        return self.Dataset.GetFeatureValue(self.Feature, instance)

    def GetClassValue(self, instance):
        return self.Dataset.GetClasses().index(instance[self.Dataset.GetClassIdx()])

    def GetSplit(self, index):
        iter = 0
        while iter <= index and FindNext():
            iter += 1


class NumericSplitIterator(SplitIterator):

    def __init__(self, dataset, feature):
        super().__init__(dataset, feature)
        self.__cuttingStrategy = None
        self.__currentIndex = 0
        self.__lastClassValue = 0
        self.__sortedInstances = None
        self.__selectorFeatureValue = 0

    def Initialize(self, instances):
        super().Initialize(instances)

        self._initialized = True

        if self.Dataset.IsNominalFeature(self.Feature):
            raise Exception("Cannot use this iterator on non-numeric feature")

        if self.Feature[1].lower() in ["integer"]:
            self.__cuttingStrategy = self.NumericOnPoint
        elif self.Feature[1].lower() in ["real", "numeric"]:
            self.__cuttingStrategy = self.NumericCenterBetweenPoints
        else:
            raise Exception(
                f"Feature type {self.Feature[1]} is not considered")

        instList = list(instances)
        filteredInsts = list(
            filter(lambda element: not self.IsMissing(element[0]), instList))
        sortedInsts = sorted(filteredInsts, key=lambda element: element[0][self.GetFeatureIdx()])
        self.__sortedInstances = sortedInsts

        self.CurrentDistribution[0] = [0]*self._numClasses
        self.CurrentDistribution[1] = FindDistribution(
            self.__sortedInstances, self.Model, self.Dataset.Class)


        if (len(self.__sortedInstances) == 0):
            return

        self.__currentIndex = -1
        self.__lastClassValue = self.FindNextClass(0)

    def FindNext(self):
        super().FindNext()
        if (self.__currentIndex >= len(self.__sortedInstances) - 1):
            return False

        self.__currentIndex += 1
        while self.__currentIndex < len(self.__sortedInstances) - 1:
            instance = self.__sortedInstances[self.__currentIndex][0]
            objClass = self.GetClassValue(instance)
            self.CurrentDistribution[0][objClass] += self.__sortedInstances[self.__currentIndex][1]
            self.CurrentDistribution[1][objClass] -= self.__sortedInstances[self.__currentIndex][1]

            if self.GetFeatureValue(instance) != self.GetFeatureValue(self.__sortedInstances[self.__currentIndex+1][0]):
                nextClassValue = self.FindNextClass(self.__currentIndex + 1)
                if (self.__lastClassValue != nextClassValue) or (self.__lastClassValue == -1 and nextClassValue == -1):
                    self.__selectorFeatureValue = self.__cuttingStrategy(
                        instance)
                    self.__lastClassValue = nextClassValue
                    return True
            self.__currentIndex += 1
        return False

    def CreateCurrentChildSelector(self):
        selector = CutPointSelector(self.Dataset, self.Feature)
        selector.CutPoint = self.__selectorFeatureValue
        return selector

    def FindNextClass(self, index):
        currentClass = self.GetClassValue(self.__sortedInstances[index][0])
        currentValue = self.GetFeatureValue(self.__sortedInstances[index][0])
        index += 1
        while index < len(self.__sortedInstances) and currentValue == self.GetFeatureValue(self.__sortedInstances[index][0]):
            if currentClass != self.GetClassValue(self.__sortedInstances[index][0]):
                return -1
            index += 1
        return currentClass

    def NumericOnPoint(self, instance):
        return instance[self.GetFeatureIdx()]

    def NumericCenterBetweenPoints(self, instance):
        return (instance[self.GetFeatureIdx()] + self.__sortedInstances[self.__currentIndex+1][0][self.GetFeatureIdx()]) / 2


class NominalSplitIterator(SplitIterator):

    def __init__(self, dataset, feature):
        super().__init__(dataset, feature)
        self.__perValueDistribution = None
        self.__totalDistribution = None
        self.__valuesCount = None
        self.__existingValues = None
        self.__iteratingTwoValues = None
        self.__valueIndex = None
        self.__twoValuesIterated = None

    def Initialize(self, instances):
        super().Initialize(instances)
        self.__perValueDistribution = {}
        self.__totalDistribution = [0]*self._numClasses

        for instance in instances:
            #print(f"instanceElem: {instance[0]}")
            if self.IsMissing(instance[0]):
                #print(f"instanceElem: {instance[0]}")
                continue
            value = self.GetFeatureValue(instance[0])
            current = [0]*self._numClasses
            if not value in self.__perValueDistribution:
                self.__perValueDistribution.update({value: current})

            classIdx = self.GetClassValue(instance[0])
            self.__perValueDistribution[value][classIdx] += instance[1]
            self.__totalDistribution[classIdx] += instance[1]

        self.__valuesCount = len(self.__perValueDistribution)
        self.__existingValues = list(self.__perValueDistribution.keys())
        self.__iteratingTwoValues = (self.__valuesCount == 2)
        self.__valueIndex = -1
        self.__twoValuesIterated = False

    def FindNext(self):
        super().FindNext()
        if self.__valuesCount == self._instances:
            return False
        if self.__iteratingTwoValues:
            if self.__twoValuesIterated:
                return False
            self.__twoValuesIterated = True
            self.CalculateCurrent(
                self.__perValueDistribution[self.__existingValues[0]])
            return True
        else:
            if(self.__valuesCount < 2 or self.__valueIndex >= self.__valuesCount - 1):
                return False
            self.__valueIndex += 1
            self.CalculateCurrent(
                self.__perValueDistribution[self.__existingValues[self.__valueIndex]])
            return True

    def CreateCurrentChildSelector(self):
        if self.__iteratingTwoValues:
            selector = MultipleValuesSelector(self.Dataset, self.Feature)
            selector.Values = list(self.__perValueDistribution.keys())
        else:
            selector = ValueAndComplementSelector(self.Dataset, self.Feature)
            selector.Value = self.__existingValues[self.__valueIndex]
        return selector


    def CalculateCurrent(self, current):
        self.CurrentDistribution[0] = current
        self.CurrentDistribution[1] = Substract(
            self.__totalDistribution, current)

class MultivariateSplitIteratorProvider(SplitIteratorProvider):
    def __init__(self, dataset):
        self.Dataset = dataset

    def GetMultivariateSplitIterator(self, features, wMin):
        result = MultivariateOrderedFeatureSplitIterator(
            self.Dataset, features)
        result.WMin = wMin
        return result


class MultivariateSplitIterator(SplitIterator):
    def __init__(self, dataset, features):
        super().__init__(dataset, None)
        self.Features = features

    def Initialize(self, instances):
        raise Exception("Must initialize as multivariate")

    def InitializeMultivariate(self, instances, node):
        if not self.Model:
            raise Exception("Model is null")
        if self.Class[1] in ['numeric', 'real', 'integer', 'string']:
            raise Exception("Cannot use this iterator on non-nominal class")
        if any(self.Dataset.IsNominalFeature(feature) for feature in self.Features):
            raise Exception("Cannot use this iterator on numeric features")

        self._numClasses = len(self.Dataset.GetClasses())
        self._instances = len(instances)
        self.CurrentDistribution = [[] for i in range(2)]
        self._initialized = True

    def FindNext(self):
        super().FindNext()


class MultivariateOrderedFeatureSplitIterator(MultivariateSplitIterator):
    def __init__(self, dataset, features):
        super().__init__(dataset, features)
        self.__filteredInstances = None
        self.__projections = None
        self.__currentIndex = 0
        self.__lastClassValue = None
        self.__sortedInstances = None
        self.__cuttingStrategy = None
        self.__selectorFeatureValue = None
        self.__weights = None
        self.WMin = None

    def InitializeMultivariate(self, instances, node):
        super().InitializeMultivariate(instances, node)

        if not self.__cuttingStrategy:
            self.__cuttingStrategy = self.NumericOnPoint
        else:
            self.__cuttingStrategy = self.NumericCenterBetweenPoints

        self.__filteredInstances = list(
            filter(lambda instance: not any(self.Dataset.IsMissing(feature, instance[0]) for feature in self.Features), instances))

        self.__projections = self.GetProjections(self.__filteredInstances)

        if not self.__projections or len(self.__projections) == 0:
            return False

        self.__sortedInstances = list()

        self.__sortedInstances = [(self.__filteredInstances[i][0], self.__filteredInstances[i][1],
                                   self.__projections[i]) for i in range(len(self.__filteredInstances))]
        self.__sortedInstances.sort(key=lambda instance: instance[2])

        self.CurrentDistribution[0] = [0]*self._numClasses
        self.CurrentDistribution[1] = FindDistribution(
            self.__sortedInstances, self.Model, self.Dataset.Class)

        self.__currentIndex = -1
        self.__lastClassValue = self.FindNextClass(0)
        return True

    def FindNext(self):
        super().FindNext()
        if (self.__currentIndex >= len(self.__sortedInstances) - 1):
            return False

        self.__currentIndex += 1
        while self.__currentIndex < len(self.__sortedInstances) - 1:
            instance = self.__sortedInstances[self.__currentIndex][0]
            value = self.__sortedInstances[self.__currentIndex][2]
            objClass = self.GetClassValue(instance)

            self.CurrentDistribution[0][objClass] += self.__sortedInstances[self.__currentIndex][1]
            self.CurrentDistribution[1][objClass] -= self.__sortedInstances[self.__currentIndex][1]

            if value != self.__sortedInstances[self.__currentIndex+1][2]:
                nextClassValue = self.FindNextClass(self.__currentIndex + 1)
                if (self.__lastClassValue != nextClassValue) or (self.__lastClassValue == -1 and nextClassValue == -1):
                    self.__selectorFeatureValue = self.__cuttingStrategy(value)
                    self.__lastClassValue = nextClassValue
                    return True
            self.__currentIndex += 1
        return False

    def CreateCurrentChildSelector(self):
        selector = MultivariateCutPointSelector(self.Dataset, self.Features)
        selector.CutPoint = self.__selectorFeatureValue
        selector.Weights = self.__weights
        return selector

    def FindNextClass(self, index):
        currentClass = self.GetClassValue(self.__sortedInstances[index][0])
        currentValue = self.__sortedInstances[index][2]
        index += 1
        while index < len(self.__sortedInstances) and currentValue == self.__sortedInstances[index][2]:
            if currentClass != self.GetClassValue(self.__sortedInstances[index][0]):
                return -1
            index += 1
        return currentClass

    def GetProjections(self, instances):
        classIdx = self.Dataset.GetClassIdx()
        featuresIdxs = [self.Dataset.GetFeatureIdx(
            feature) for feature in self.Features]
        ldaData = [[instance[0][featureIdx] for featureIdx in featuresIdxs]
                   for instance in instances]
        ldaTargets = [self.Dataset.GetIndexOfValue(
            self.Class[0], instance[0][classIdx]) for instance in instances]

        lda = LDA(n_components=1)
        try:
            ldaOutput = lda.fit(ldaData, ldaTargets).transform(ldaData)

            if len(ldaOutput) == 0:
                return list()

            w = lda.coef_[0]
            if len(w) == 0:
                return list()

            self.__weights = {self.Features[i]: w[i]
                              for i in range(0, len(self.Features))}

            w_norm = math.sqrt(sum(map(lambda x: math.pow(x, 2), w)))

            for x in w:
                if (w_norm == 0):
                    print(f"x: {x} w_norm {w_norm}")
                
                divNum = abs(x/w_norm)
                if ( (not math.isnan(divNum)) and divNum < self.WMin):
                    print("x/w_norm is smaller than wMin")
                    return list()

            return list(map(lambda r: r[0], ldaOutput))

        except Exception as e:
            return list()

    def NumericOnPoint(self, value):
        return value

    def NumericCenterBetweenPoints(self, value):
        return (value + self.__sortedInstances[self.__currentIndex][2]) / 2
