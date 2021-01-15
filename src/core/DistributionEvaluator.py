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
    #parent = [0.0, 45.0, 45.0]
    #children = 2*[3*[0]]
    #children[0] = [0.0, 42.0, 2.0]
    #children[1] = [0.0, 3.0, 43.0]
    print(f"multiClass Hellinger")
    print(f"parent: {parent}")
    print(f"children: {children}")
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

            if parent[i] == 0.0:
                if children[0][i] == 0.0:
                    s1p = float('nan')
                else: s1p = float('inf')
            else: s1p = math.sqrt(children[0][i] / parent[i])
            #s1p = math.sqrt(children[0][i] / parent[i])

            s1nA = SumDifferent(children[0], i)

            if (tn == 0.0):
                if s1nA == 0.0:
                    s1n = float('nan')
                else:
                    s1n = float('inf')
            else: s1n = math.sqrt(s1nA / tn)
            #s1n = math.sqrt(SumDifferent(children[0], i) / tn)

            if parent[i] == 0.0:
                if children[1][i] == 0.0:
                    s2p = float('nan')
                else: s2p = float('inf')
            else: s2p = math.sqrt(children[1][i] / parent[i])
            #s2p = math.sqrt(children[1][i] / parent[i])

            s2nA = SumDifferent(children[1], i)
            if (tn == 0.0):
                if s2nA == 0.0:
                    s2n = float('nan')
                else:
                    s2n = float('inf')
            else: s2n = math.sqrt(s2nA / tn)
            #s2n = math.sqrt(SumDifferent(children[1], i) / tn)

            currentValue = math.pow(s1p - s1n, 2) + math.pow(s2p - s2n, 2)
            print(f"negT: {tn} posL: {s1p} negL: {s1n} posR: {s2p} negR: {s2n} curr_value: {currentValue}")
            
            if currentValue > hellinger:
                hellinger = currentValue
    except ZeroDivisionError:
        print(f"zeroDivError")

    print(f"hellingerVal {hellinger}")
    if hellinger == float('inf'):
        print(f"IsInfinity")
        return sys.float_info.max

    return math.sqrt(hellinger)

def QuinlanGain (parent, children):
    #print("I am in quinlan")
    
    #print(f"ParentQuinlan: {parent}   ChildrenQuinlan: {children}")
    #print(" ")
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
