from .DecisionTree import DecisionTree, DecisionTreeNode
from .Helpers import MultiplyBy, AddTo


class DecisionTreeClassifier(object):
    def __init__(self, tree):
        self.__DecisionTree = tree
        self.__Model = tree.Model

    @property
    def DecisionTree(self):
        return self.__DecisionTree
    @DecisionTree.setter
    def DecisionTree(self, new_decision_tree):
        self.__DecisionTree = new_decision_tree

    @property
    def Model(self):
        return self.__Model
    @Model.setter
    def Model(self, new_model):
        self.__Model = new_model

    def ClassifyInstance(self, node, instance, instanceMembership):
        if node.IsLeaf:
            print(f"node.IsLeaf: {MultiplyBy(node.Data, instanceMembership)}")
            return MultiplyBy(node.Data, instanceMembership)

        childrenSelection = node.ChildSelector.Select(instance)
        result = None
        if (childrenSelection is not None):
            if (len(childrenSelection) != len(node.Children)):
                raise Exception("Child index is out of range")

            for i in range(len(childrenSelection)):
                selection = childrenSelection[i]
                if selection > 0:
                    child = node.Children[i]
                    childValue = self.ClassifyInstance(
                        child, instance, instanceMembership)
                    if result is not None:
                        result = AddTo(result, childValue)
                    else:
                        result = childValue
            print(f"result1: {result}")
            return result

        else:
            totalNodeMembership = sum(node.Data)
            for i in range(len(node.Children)):
                child = node.Children[i]
                childMembership = sum(node.Children[i].Data)
                childValue = self.ClassifyInstance(
                    child, instance, childMembership / (totalNodeMembership * instanceMembership))
                if result is not None:
                    result = AddTo(result, childValue)
                else:
                    result = childValue
        print(f"result2: {result}")            
        return result

    def Classify(self, instance):
        classification = self.ClassifyInstance(
            self.DecisionTree.TreeRootNode, instance, 1)
        return MultiplyBy(classification, (1/sum(classification)))