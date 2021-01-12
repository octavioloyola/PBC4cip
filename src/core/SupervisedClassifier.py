from core.DecisionTree import DecisionTree, DecisionTreeNode
from core.Helpers import MultiplyBy, AddTo


class DecisionTreeClassifier(object):
    def __init__(self, tree):
        self.DecisionTree = tree
        self.Model = tree.Model

    def Classify(self, instance):
        classification = self.ClassifyInstance(
            self.DecisionTree.TreeRootNode, instance, 1)
        return MultiplyBy(classification, (1/sum(classification)))

    def ClassifyInstance(self, node, instance, instanceMembership):
        if node.IsLeaf:
            return MultiplyBy(node.Data, instanceMembership)

        childrenSelection = node.ChildSelector.Select(instance)
        result = None
        if (childrenSelection != None):
            if (len(childrenSelection) != len(node.Children)):
                raise Exception("Child index is out of range")

            for i in range(len(childrenSelection)):
                selection = childrenSelection[i]
                if selection > 0:
                    child = node.Children[i]
                    childValue = ClassifyInstance(
                        child, instance, instanceMembership)
                    if result != None:
                        result = AddTo(result, childValue)
                    else:
                        result = childValue
        else:
            totalNodeMembership = sum(node.Data)
            for i in range(len(node.Children)):
                child = node.Children[i]
                childMembership = sum(node.Children[i].Data)
                childValue = ClassifyInstance(
                    child, instance, childMembership / (totalNodeMembership * instanceMembership))
                if result != None:
                    result = AddTo(result, childValue)
                else:
                    result = childValue
                    
        return result
