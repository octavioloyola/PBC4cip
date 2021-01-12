import math
import random
from core.DecisionTree import DecisionTree, DecisionTreeNode
from core.WinningSplitSelector import WinningSplitSelector
from core.SplitIterator import SplitIterator
from core.Helpers import CreateMembershipTuple, FindDistribution
from core.SplitIterator import SplitIteratorProvider, MultivariateSplitIteratorProvider
from core.ForwardFeatureIterator import ForwardFeatureIterator


class DecisionTreeBuilder():
    def __init__(self, dataset, pruneResult=None):
        self.MinimalInstanceMembership = 0.05
        self.MinimalSplitGain = 1e-30
        self.MinimalObjByLeaf = 2
        self.MaxDepth = -1
        self.PruneResult = False
        self.Dataset = dataset
        self.FeatureCount = 0
        self.StopCondition = None
        self.DistributionEvaluator = None
        self.OnSelectingFeaturesToConsider = None
        self.SplitIteratorProvider = SplitIteratorProvider(self.Dataset)

    def Build(self):
        if self.MinimalSplitGain <= 0:
            self.MinimalSplitGain = 1e-30

        #print(f"MinimalSplitGain{self.MinimalSplitGain}")
        currentContext = []

        #print(f"model: {self.Dataset.Model}")

        objectMebership = CreateMembershipTuple(self.Dataset.Instances)
        #print(f"objectMebership: {objectMebership}")
        classFeature = self.Dataset.Class

        result = DecisionTree(self.Dataset)

        filteredObjMembership = tuple(
            filter(lambda x: x[1] >= self.MinimalInstanceMembership, objectMebership))
        print(f"filteredObj: {len(filteredObjMembership)}")
        print(f"tuples[0]: {filteredObjMembership[0]}")
        print(f"tuples[50]: {filteredObjMembership[50]}")
        print(f"tuples[134]: {filteredObjMembership[134]}")

        parentDistribution = FindDistribution(
            filteredObjMembership, self.Dataset.Model, self.Dataset.Class)
        print(f"parentDistribution: {parentDistribution}" )

        result.TreeRootNode = DecisionTreeNode()
        result.TreeRootNode.Data = parentDistribution
        print("About to fill Node")
        self.FillNode(result.TreeRootNode,
                      filteredObjMembership, 0, currentContext)
        return result

    #
    def FillNode(self, node, instanceTuples, level, currentContext):
        #print("fillingNodes")
        if self.StopCondition(node.Data, self.Dataset.Model, self.Dataset.Class):
            return
        if self.MaxDepth >= 0 and (level >= self.MaxDepth - 1):
            return
        if sum(node.Data) <= self.MinimalObjByLeaf:
            return
        

        whichBetterToFind = 1
        winningSplitSelector = WinningSplitSelector(whichBetterToFind)
        currentGain = 0

        sampleFeatures = self.OnSelectingFeaturesToConsider(
            list(map(lambda attribute: attribute[0], self.Dataset.Attributes)), self.FeatureCount)
        #print(f"SampleFeatures {sampleFeatures} and self.Dataset.Class[0]: {self.Dataset.Class}")
        for feature in sampleFeatures:
            #if feature != self.Dataset.Class[0]:
                splitIterator = self.SplitIteratorProvider.GetSplitIterator(
                    feature)
                splitIterator.Initialize(instanceTuples)
                #print(f"splitIterator._instances: {splitIterator._instances}")
                while splitIterator.FindNext():
                    currentGain = self.DistributionEvaluator(
                        node.Data, splitIterator.CurrentDistribution)
                    #print(f"currentGainUni {currentGain}")
                    if currentGain >= self.MinimalSplitGain:
                        winningSplitSelector.EvaluateThis(
                            currentGain, splitIterator, level) #currentGain y splitIterator es iwal
        print(f"WinningSplitSelector {winningSplitSelector.List} \n")
        if winningSplitSelector.IsWinner():
            maxSelector = winningSplitSelector.WinningSelector
            node.ChildSelector = maxSelector
            node.Children = list()
            instancesPerChildNode = CreateChildrenInstances(
                instanceTuples, maxSelector, self.MinimalInstanceMembership)

            #print(f"childrenCount {maxSelector.ChildrenCount}")
            for index in range(maxSelector.ChildrenCount):
                childNode = DecisionTreeNode()
                childNode.Parent = node
                node.Children.append(childNode)
                childNode.Data = winningSplitSelector.WinningDistribution[index]

                self.FillNode(
                    childNode, instancesPerChildNode[index], level + 1, currentContext)

        #print(f"WinningSplitSelectorDistribution: {winningSplitSelector.WinningSelectorList}")
        #print(f"WinningSplit: {winningSplitSelector.List}")
        return


def CreateChildrenInstances(instances, selector, threshold=None):
    if not threshold:
        threshold = 0

    result = list()
    for child in range(selector.ChildrenCount):
        result.append(list(tuple()))

    for instance in instances:
        selection = selector.Select(instance[0])
        if selection:
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
    def __init__(self, dataset, pruneResult=None):
        super().__init__(dataset, pruneResult)
        self.MinimalForwardGain = 0
        self.WMin = 0  # Minimal absolute value for each weight after normalizing
        self.SplitIteratorProvider = MultivariateSplitIteratorProvider(
            self.Dataset)

    def Build(self):
        if self.MinimalSplitGain <= 0:
            self.MinimalSplitGain = 1e-30

        #print(f"MinimalSplitGain: { self.MinimalSplitGain }")
        currentContext = []

        objectMebership = CreateMembershipTuple(self.Dataset.Instances)
        classFeature = self.Dataset.Class

        result = DecisionTree(self.Dataset)

        filteredObjMembership = tuple(
            filter(lambda x: x[1] >= self.MinimalInstanceMembership, objectMebership))

        parentDistribution = FindDistribution(
            filteredObjMembership, self.Dataset.Model, self.Dataset.Class)

        result.TreeRootNode = DecisionTreeNode()
        result.TreeRootNode.Data = parentDistribution

        self.FillNode(result.TreeRootNode,
                      filteredObjMembership, 0, currentContext)

        return result

    #
    def FillNode(self, node, instanceTuples, level, currentContext):
        print("fillNode")
        if self.StopCondition(node.Data, self.Dataset.Model, self.Dataset.Class):
            return
        if self.MaxDepth >= 0 and level >= self.MaxDepth - 1:
            return
        if sum(node.Data) <= self.MinimalObjByLeaf:
            return

        whichBetterToFind = 1
        winningSplitSelector = WinningSplitSelector(whichBetterToFind)
        currentGain = 0

        sampleFeatures = self.OnSelectingFeaturesToConsider(
            list(map(lambda attribute: attribute[0], self.Dataset.Attributes)), self.FeatureCount)

        bestFeature = None

        print(f"sampleFeatures: {sampleFeatures}")
        for feature in sampleFeatures:
            splitIterator = self.SplitIteratorProvider.GetSplitIterator(
                feature)
            if not splitIterator:
                raise Exception(f"Undefined iterator for feature {feature}")
            splitIterator.Initialize(instanceTuples)
            while splitIterator.FindNext():
                currentGain = self.DistributionEvaluator(
                    node.Data, splitIterator.CurrentDistribution)
                #print(f"currentGainMulti: " + currentGain)
                if currentGain >= self.MinimalSplitGain:
                    if winningSplitSelector.EvaluateThis(
                            currentGain, splitIterator, level):
                        bestFeature = self.Dataset.GetAttribute(feature)

        # Forward feature selection

        if bestFeature is not None and not self.Dataset.IsNominalFeature(bestFeature):
            sampleFeatures = list(filter(lambda feature: not self.Dataset.IsNominalFeature(
                feature), [self.Dataset.GetAttribute(feature) for feature in sampleFeatures]))
            featureIterator = ForwardFeatureIterator(
                self.Dataset, sampleFeatures)
            featureIterator.Add(bestFeature)
            while featureIterator.FeaturesRemain:
                bestFeature = None
                for features in featureIterator.GetFeatures():
                    candidateFeature = features[0]

                    splitIterator = self.SplitIteratorProvider.GetMultivariateSplitIterator(
                        features, self.WMin)
                    if not splitIterator:
                        raise Exception(
                            f"Undefined iterator for features {','.join(map(lambda feature: feature[0],features))}")

                    valid = splitIterator.InitializeMultivariate(
                        instanceTuples, node)

                    if not valid:
                        break

                    while splitIterator.FindNext():
                        currentGain = self.DistributionEvaluator(
                            node.Data, splitIterator.CurrentDistribution)
                        if currentGain >= self.MinimalSplitGain and (currentGain - winningSplitSelector.MinStoredValue) >= self.MinimalForwardGain:
                            if winningSplitSelector.EvaluateThis(currentGain, splitIterator, level):
                                bestFeature = candidateFeature
                if not bestFeature:
                    break
                else:
                    featureIterator.Add(bestFeature)

        if winningSplitSelector.IsWinner():
            maxSelector = winningSplitSelector.WinningSelector
            node.ChildSelector = maxSelector
            node.Children = list()
            instancesPerChildNode = CreateChildrenInstances(
                instanceTuples, maxSelector, self.MinimalInstanceMembership)

            for index in range(maxSelector.ChildrenCount):
                childNode = DecisionTreeNode()
                childNode.Parent = node
                childNode.Data = winningSplitSelector.WinningDistribution[index]
                node.Children.append(childNode)

                self.FillNode(
                    childNode, instancesPerChildNode[index], level + 1, currentContext)
        
        print(f"WinningSplitSelectorDistribution: {winningSplitSelector.WinningDistribution}")

        return
