import math
import sys


def Hellinger(parent, children):
    print(f"In hellinger")
    #print(f"parent: {type(parent)}")
    #print(parent)
    #print(f"children: {type(children)}")
    #print(children)
    # region Preconditions
    if len(children) != 2:
        raise Exception(
            "Hellinger Distance need only two child nodes (binary split)")

    if sum(parent) == max(parent):
        return 0
    # endregion
    #print(f"children:[0][0] {children[0][0]}  parent[0] {parent[0]}")
    if (parent[0] != 0 and parent[1] != 0):
        s1p = math.sqrt(children[0][0] / parent[0])
        s1n = math.sqrt(children[0][1] / parent[1])

        s2p = math.sqrt(children[1][0] / parent[0])
        s2n = math.sqrt(children[1][1] / parent[1])

        result = math.sqrt(math.pow(s1p - s1n, 2) + math.pow(s2p - s2n, 2))
        #print(f"result: {result}")

        return result
    else:
         return 0


def MultiClassHellinger(parent, children):
    print(f"multiClass Hellinger")
    # region Preconditions
    if len(children) != 2:
        raise Exception(
            "Hellinger Distance need only two child nodes (binary split)")

    if sum(parent) == max(parent):
        return 0
    # endregion

    hellinger = sys.float_info.min

    try:
        for i in range(len(parent)):
            tn = SumDifferent(parent, i)

            s1p = math.sqrt(children[0][i] / parent[i])
            s1n = math.sqrt(SumDifferent(children[0], i) / tn)
            s2p = math.sqrt(children[1][i] / parent[i])
            s2n = math.sqrt(SumDifferent(children[1], i) / tn)

            currentValue = math.pow(s1p - s1n, 2) + math.pow(s2p - s2n, 2)
            if currentValue > hellinger:
                hellinger = currentValue
    except ZeroDivisionError:
        return sys.float_info.max

    return math.sqrt(hellinger)

def QuinlanGain (parent, children):
    #print("I am in quinlan")
    print(children)
    #rint(" ")
    #parent = [45.0, 45.0, 45.0]
    #children = 2*[3*[0]]
    #children[0] = [44.0, 20.0, 2.0]
    #children[1] = [1.0, 25.0, 43.0]
    result = GetImpurity(parent)
    total = sum(parent)
    nonMissing = 0.0

    for distribution in children:
        childCount = sum(distribution)
        nonMissing += childCount
        result -= GetImpurity(distribution) * (childCount * 1.0 / total)

    #print(f"result: {result * (nonMissing) / total }")
    return result * (nonMissing) / total 


def SumDifferent(vector, index):
    sumValue = 0
    for i in range(len(vector)):
        if index != i:
            sumValue += vector[i]
    return sumValue

def GetImpurity (distribution) :
    #print(f"dist: {distribution}")
    result = 0
    count = sum(distribution)

    for value in distribution:
        if (value != 0):
            p = value * 1.0 / count
            result -= p * math.log(p, 2)

    #print(f"result: {result}")
    return result
