from copy import copy
from .Item import SubsetRelation
from .FilteredCollection import FilteredCollection
from .DecisionTreeBuilder import SelectorContext
from .Item import CutPointBasedBuilder, MultipleValuesBasedBuilder, ValueAndComplementBasedBuilder, MultivariateCutPointBasedBuilder
from .FeatureSelectors import CutPointSelector, MultipleValuesSelector, ValueAndComplementSelector, MultivariateCutPointSelector
from itertools import chain
from collections import OrderedDict


class EmergingPattern(object):
    def __init__(self, dataset, items=None):
        self.__Dataset = dataset
        self.__Model = self.Dataset.Model
        self.__Items = None
        if not items:
            self.Items = list()
        else:
            self.Items = items
        self.__Counts = []
        self.__Supports = []
    
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
    def Items(self):
        return self.__Items
    @Items.setter
    def Items(self, new_items):
        self.__Items = new_items
    
    @property
    def Counts(self):
        return self.__Counts
    @Counts.setter
    def Counts(self, new_counts):
        self.__Counts = new_counts
    
    @property
    def Supports(self):
        return self.__Supports
    @Supports.setter
    def Supports(self, new_supports):
        self.__Supports = new_supports

    def IsMatch(self, instance):
        for item in self.Items:
            if not item.IsMatch(instance):
                return False
        return True

    def __UpdateCountsAndSupport(self, instances):
        matchesCount = [0]*len(self.Dataset.Class[1])

        for instance in instances:
            if(self.IsMatch(instance)):
                matchesCount[instance[self.Dataset.GetClassIdx()]] += 1
        self.Counts = matchesCount
        self.Supports = self.CalculateSupports(matchesCount)

    def CalculateSupports(self, data, classFeatureParam=None):
        if classFeatureParam == None:
            classInfo = self.Dataset.ClassInformation
            result = copy(data)
            for i in range(len(result)):
                if classInfo.Distribution[i] != 0:
                    result[i] /= classInfo.Distribution[i]
                else:
                    result[i] = 0
            return result
        else:
            classFeature = self.Dataset.ClassInformation.Feature
            featureInformation = self.Dataset.ClassInformation.Distribution
            result = copy(data)
            for i in range(len(result)):
                if featureInformation[i] != 0:
                    result[i] /= featureInformation[i]
                else:
                    result[i] = 0
            return result
                

    def __Clone(self):

        result = EmergingPattern(self.Dataset, self.Items)
        result.Counts = copy(self.Counts)
        result.Supports = copy(self.Supports)
        return result

    def __repr__(self):
        return self.BaseRepresentation() + "\n" + self.SupportInfo()

    def BaseRepresentation(self):
        return ' AND '.join(map(lambda item: item.__repr__(), self.Items))

    def SupportInfo(self):
        return ' '.join(map(lambda count, support, className: f"{className} count: {str(count)} support: {str(round(support,2) * 100)}% ", self.Counts, self.Supports, self.Dataset.Class[1]))

    def ToString(self):
        dictOfPatterns = {"Pattern": self.BaseRepresentation()}

        dictOfClasses = {self.Dataset.Class[1][i]+" Count": self.Counts[i]
                         for i in range(0, len(self.Dataset.Class[1]))}

        dictOfClasses.update({self.Dataset.Class[1][i]+" Support": self.Supports[i]
                              for i in range(0, len(self.Dataset.Class[1]))})

        for key in sorted(dictOfClasses.keys()):
            dictOfPatterns.update({key: dictOfClasses[key]})

        return dictOfPatterns


class EmergingPatternCreator(object):
    def __init__(self, dataset):
        self.__Dataset = dataset
        self.__builderForType = {
            CutPointSelector: CutPointBasedBuilder,
            MultipleValuesSelector: MultipleValuesBasedBuilder,
            ValueAndComplementSelector: ValueAndComplementBasedBuilder,
            MultivariateCutPointSelector: MultivariateCutPointBasedBuilder
        }
    
    @property
    def Dataset(self):
        return self.__Dataset
    @Dataset.setter
    def Dataset(self, new_dataset):
        self.__Dataset = new_dataset

    def __Create(self, contexts):
        pattern = EmergingPattern(self.Dataset)
        for context in contexts:
            childSelector = context.Selector
            builder = self.__builderForType[context.Selector.__class__]()
            item = builder.GetItem(childSelector, context.Index)
            pattern.Items.append(item)
        return pattern

    def ExtractPatterns(self, treeClassifier, patternFound):
        context = list()
        self.__DoExtractPatterns(
            treeClassifier.DecisionTree.TreeRootNode, context, patternFound)

    def __DoExtractPatterns(self, node, contexts, patternFound):
        if node.IsLeaf:
            newPattern = self.__Create(contexts)
            newPattern.Counts = node.Data
            newPattern.Supports = newPattern.CalculateSupports(node.Data)
            if patternFound is not None:
                patternFound(newPattern)
        else:
            for index in range(len(node.Children)):
                selectorContext = SelectorContext()
                selectorContext.Index = index
                selectorContext.Selector = node.ChildSelector
                context = selectorContext

                contexts.append(context)
                self.__DoExtractPatterns(node.Children[index], contexts, patternFound)
                contexts.remove(context)


class EmergingPatternComparer(object):
    def __init__(self, itemComparer):
        self.__Comparer = itemComparer
    @property
    def Comparer(self):
        return self.__Comparer
    @Comparer.setter
    def Comparer(self, new_comparer):
        self.__Comparer = new_comparer

    def Compare(self, leftPattern, rightPattern):
        directSubset = self.IsSubset(leftPattern, rightPattern)
        inverseSubset = self.IsSubset(rightPattern, leftPattern)
        if (directSubset and inverseSubset):
            return SubsetRelation.Equal
        elif (directSubset):
            return SubsetRelation.Subset
        elif (inverseSubset):
            return SubsetRelation.Superset
        else:
            return SubsetRelation.Unrelated

    def IsSubset(self, pat1, pat2):
        def f(x, y):
            relation = self.Comparer.Compare(self, x, y)
            return relation == SubsetRelation.Equal or relation == SubsetRelation.Subset

        for x in pat2.Items:
            all_bool = False
            for y in pat1.Items:
                if f(y, x):
                    all_bool = True
                    break
            if not all_bool:
                return False
        return True

class EmergingPatternSimplifier(object):
    def __init__(self, itemComparer):
        self.__comparer = itemComparer
        self.__collection = FilteredCollection(
            self.__comparer, SubsetRelation.Subset)

    def Simplify(self, pattern):
        resultPattern = EmergingPattern(pattern.Dataset)
        resultPattern.Counts = copy(pattern.Counts)
        resultPattern.Supports = copy(pattern.Supports)
        self.__collection.current = resultPattern.Items
        self.__collection.AddRange(pattern.Items)
        return resultPattern
