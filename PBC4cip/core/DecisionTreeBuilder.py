import math
import random
from .DecisionTree import DecisionTree, DecisionTreeNode
from .WinningSplitSelector import WinningSplitSelector
from .SplitIterator import SplitIterator
from .Helpers import CreateMembershipTupleList, FindDistribution, combine_instances
from .SplitIterator import SplitIteratorProvider, MultivariateSplitIteratorProvider
from .ForwardFeatureIterator import ForwardFeatureIterator
from .DistributionTester import PureNodeStopCondition, AlwaysTrue

class DecisionTreeBuilder():
    def __init__(self, dataset, X, y):
        self.__MinimalInstanceMembership = 0.05
        self.__MinimalSplitGain = 1e-30
        self.__MinimalObjByLeaf = 2
        self.__MaxDepth = -1
        self.__PruneResult = False
        self.__Dataset = dataset
        self.__trainInstances = combine_instances(X, y)
        self.__FeatureCount = 0
        self.__StopCondition = PureNodeStopCondition
        self.__distributionEvaluator = None
        self.__OnSelectingFeaturesToConsider = None
        self.__SplitIteratorProvider = SplitIteratorProvider(self.Dataset)
    
    @property
    def MinimalInstanceMembership(self):
        return self.__MinimalInstanceMembership
    
    @property
    def StopCondition(self):
        return self.__StopCondition
    
    @property
    def MaxDepth(self):
        return self.__MaxDepth
    
    @property
    def MinimalObjByLeaf(self):
        return self.__MinimalObjByLeaf
    
    @property
    def SplitIteratorProvider(self):
        return self.__SplitIteratorProvider
    @SplitIteratorProvider.setter
    def SplitIteratorProvider(self, new_split_iterator_provider):
        self.__SplitIteratorProvider = new_split_iterator_provider
    
    @property
    def MinimalSplitGain(self):
        return self.__MinimalSplitGain
    @MinimalSplitGain.setter
    def MinimalSplitGain(self, new_minimal_split_gain):
        self.__MinimalSplitGain = new_minimal_split_gain

    @property
    def Dataset(self):
        return self.__Dataset

    @property
    def trainInstances(self):
        return self.__trainInstances
    @trainInstances.setter
    def trainInstances(self, new_train_instances):
        self.__trainInstances = new_train_instances
    
    @property
    def distributionEvaluator(self):
        return self._distributionEvaluator

    @distributionEvaluator.setter
    def distributionEvaluator(self, new_distributionEvaluator):
        self._distributionEvaluator = new_distributionEvaluator

    def Build(self):
        if self.MinimalSplitGain <= 0:
            raise Exception(f"MinimalSplitGain err in Build UniVariate")
            self.MinimalSplitGain = 1e-30

        currentContext = []
        objectMebership = CreateMembershipTupleList(self.trainInstances)
        classFeature = self.Dataset.Class
        result = DecisionTree(self.Dataset)

        filteredObjMembership = list(
            filter(lambda x: x[1] >= self.MinimalInstanceMembership, objectMebership))

        parentDistribution = FindDistribution(
            filteredObjMembership, self.Dataset.Model, self.Dataset.Class)

        result.TreeRootNode = DecisionTreeNode(parentDistribution)
        self.__FillNode(result.TreeRootNode,
                      filteredObjMembership, 0, currentContext)
        return result

    def __FillNode(self, node, instanceTuples, level, currentContext):
        if self.StopCondition(node.Data, self.Dataset.Model, self.Dataset.Class):
            return
        if self.MaxDepth >= 0 and (level >= self.MaxDepth - 1):
            return
        if sum(node.Data) <= self.MinimalObjByLeaf:
            return
        
        whichBetterToFind = 1
        winningSplitSelector = WinningSplitSelector(whichBetterToFind)
        
        sampleFeatures = self.OnSelectingFeaturesToConsider(
            list(map(lambda attribute: attribute[0], self.Dataset.Attributes)), self.FeatureCount)

        for feature in sampleFeatures:
            if feature != self.Dataset.Class[0]:
                splitIterator = self.SplitIteratorProvider.GetSplitIterator(feature)
                splitIterator.Initialize(instanceTuples)
                while splitIterator.FindNext():
                    currentGain = self._distributionEvaluator(
                        node.Data, splitIterator.CurrentDistribution)
                    if currentGain >= self.MinimalSplitGain:
                        winningSplitSelector.EvaluateThis(
                            currentGain, splitIterator, level)
        if winningSplitSelector.IsWinner():
            maxSelector = winningSplitSelector.WinningSelector
            node.ChildSelector = maxSelector
            node.Children = list()
            instancesPerChildNode = CreateChildrenInstances(
                instanceTuples, maxSelector, self.MinimalInstanceMembership)

            for index in range(maxSelector.ChildrenCount):
                childNode = DecisionTreeNode(winningSplitSelector.WinningDistribution[index])
                childNode.Parent = node
                node.Children.append(childNode)

                self.__FillNode(
                    childNode, instancesPerChildNode[index], level + 1, currentContext)

        return


def CreateChildrenInstances(instances, selector, threshold):

    result = list()
    for child in range(selector.ChildrenCount):
        result.append(list(tuple()))

    for instance in instances:
        selection = selector.Select(instance[0])
        if selection is not None:
            for index in range(len(selection)):
                if selection[index] > 0:
                    newMembership = selection[index] * instance[1]
                    if newMembership >= threshold:
                        result[index].append(
                            tuple((instance[0], newMembership)))

    return result



class SelectorContext():
    def __init__(self):
        self.Index = 0
        self.Selector = None


class MultivariateDecisionTreeBuilder(DecisionTreeBuilder):
    def __init__(self, dataset, X, y):
        super().__init__(dataset, X, y)
        self.MinimalForwardGain = 0
        #self.__trainInstances = combine_instances(X, y)
        self.WMin = 0  # Minimal absolute value for each weight after normalizing
        self.SplitIteratorProvider = MultivariateSplitIteratorProvider(self.Dataset)
    
    

    def Build(self):
        if self.MinimalSplitGain <= 0:
            self.MinimalSplitGain = 1e-30

        currentContext = []

        objectMebership = CreateMembershipTupleList(self.trainInstances)

        classFeature = self.Dataset.Class

        result = DecisionTree(self.Dataset)

        filteredObjMembership = list(
            filter(lambda x: x[1] >= self.MinimalInstanceMembership, objectMebership))
        
        parentDistribution = FindDistribution(
            filteredObjMembership, self.Dataset.Model, self.Dataset.Class)
        print(f"ParentDist: {parentDistribution}")

        result.TreeRootNode = DecisionTreeNode(parentDistribution)

        self.__FillNode(result.TreeRootNode,
                      filteredObjMembership, 0, currentContext)

        return result

    def __FillNode(self, node, instanceTuples, level, currentContext):
        print(f"A node is being filled: context: {currentContext} len: {len(currentContext)}")
        print(f"Data of the node: {node} ")
        if self.StopCondition(node.Data, self.Dataset.Model, self.Dataset.Class):
            print(f"returned FillNode stopCondition")
            return
        if self.MaxDepth >= 0 and level >= self.MaxDepth - 1:
            print(f"return FillNode MaxDepth")
            return
        if sum(node.Data) <= self.MinimalObjByLeaf:
            print(f"return FillNode MinimalObjByLeaf")
            return

        whichBetterToFind = 1
        winningSplitSelector = WinningSplitSelector(whichBetterToFind)
        currentGain = 0
        sampleFeatures = self.OnSelectingFeaturesToConsider(
            list(map(lambda attribute: attribute[0], self.Dataset.Attributes)), self.FeatureCount)

        bestFeature = None

        for feature in sampleFeatures:
            if node.Data[0] == 124.0:
                print(f"124 feature: {feature}")
            splitIterator = self.SplitIteratorProvider.GetSplitIterator(
                feature)
            #print(f"splitIterator: {splitIterator}")
            if not splitIterator:
                raise Exception(f"Undefined iterator for feature {feature}")
            #print(f"ABout to init splitIterator")
            if node.Data[0] == 124.0:
                print(f"instanceTuples: {instanceTuples}")
            splitIterator.Initialize(instanceTuples)
            while splitIterator.FindNext():
                currentGain = self._distributionEvaluator(node.Data, splitIterator.CurrentDistribution)
                if node.Data[0] == 124.0:
                    print(f"124 currentGain: {currentGain}")
                #print(f"currentGain: {currentGain}")
                if currentGain >= self.MinimalSplitGain:
                    if winningSplitSelector.EvaluateThis(
                            currentGain, splitIterator, level):
                        bestFeature = self.Dataset.GetAttribute(feature)
                        print(f"best feature: {bestFeature}")


        if bestFeature is not None and not self.Dataset.IsNominalFeature(bestFeature):
            print(f"\n\n Forward Feature selec: feat: {bestFeature}")
            sampleFeatures = list(filter(lambda feature: not self.Dataset.IsNominalFeature(
                feature), [self.Dataset.GetAttribute(feature) for feature in sampleFeatures]))
            print(f"sampleFeatures: {sampleFeatures}")
            featureIterator = ForwardFeatureIterator(
                self.Dataset, sampleFeatures)
            featureIterator.Add(bestFeature)
            print(f"candidateFeaturesSize: {len(featureIterator.CandidateFeatures)}")
            while featureIterator.FeaturesRemain:
                bestFeature = None
                for features in featureIterator.GetFeatures():
                    candidateFeature = features[0]
                    print(f"\ncandidateFeat: {candidateFeature}")

                    splitIterator = self.SplitIteratorProvider.GetMultivariateSplitIterator(
                        features, self.WMin)
                    if not splitIterator:
                        raise Exception(
                            f"Undefined iterator for features {','.join(map(lambda feature: feature[0],features))}")

                    print(f"About to initialize multivariate splitIterator")
                    valid = splitIterator.InitializeMultivariate(
                        instanceTuples, node)
                    print(f"valid: {valid}")

                    if not valid:
                        break

                    while splitIterator.FindNext():
                        currentGain = self._distributionEvaluator(
                            node.Data, splitIterator.CurrentDistribution)
                        print(f"currentGain ForwardFeature: {currentGain}")
                        if currentGain >= self.MinimalSplitGain and (currentGain - winningSplitSelector.MinStoredValue) >= self.MinimalForwardGain:
                            if winningSplitSelector.EvaluateThis(currentGain, splitIterator, level):
                                print(f"bestFeature is {candidateFeature}")
                                bestFeature = candidateFeature
                        #else:
                            #print(f"currGain: {currentGain} minStored: {winningSplitSelector.MinStoredValue}= {currentGain - winningSplitSelector.MinStoredValue} >= {self.MinimalForwardGain}")
                if bestFeature is None:
                    print(f"bestFeature: {bestFeature} break")
                    break
                else:
                    featureIterator.Add(bestFeature)

        if winningSplitSelector.IsWinner():
            maxSelector = winningSplitSelector.WinningSelector
            node.ChildSelector = maxSelector
            node.Children = list()
            print(f"maxSelectorChildrenCount: {maxSelector.ChildrenCount}")
            instancesPerChildNode = CreateChildrenInstances(
                instanceTuples, maxSelector, self.MinimalInstanceMembership)

            for index in range(maxSelector.ChildrenCount):
                childNode = DecisionTreeNode(winningSplitSelector.WinningDistribution[index])
                childNode.Parent = node
                node.Children.append(childNode)

                self.__FillNode(
                    childNode, instancesPerChildNode[index], level + 1, currentContext)
        

        return
