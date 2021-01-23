import math
import sys


def Hellinger(parent, children):
    if len(children) != 2:
        raise Exception(
            "Hellinger Distance needs only two child nodes (binary split)")

    if sum(parent) == max(parent):
        return 0
    if (parent[0] != 0 and parent[1] != 0):
        s1p = math.sqrt(children[0][0] / parent[0])
        s1n = math.sqrt(children[0][1] / parent[1])

        s2p = math.sqrt(children[1][0] / parent[0])
        s2n = math.sqrt(children[1][1] / parent[1])

        result = math.sqrt(math.pow(s1p - s1n, 2) + math.pow(s2p - s2n, 2))

        return result
    else:
         return 0


def MultiClassHellinger(parent, children):
    if len(children) != 2:
        raise Exception(
            "Hellinger Distance needs only two child nodes (binary split)")

    if sum(parent) == max(parent):
        return 0

    hellinger = sys.float_info.min

    try:
        for i in range(len(parent)):
            tn = __SumDifferent(parent, i)

            if parent[i] == 0.0:
                if children[0][i] == 0.0:
                    s1p = float('nan')
                else: s1p = float('inf')
            else: s1p = math.sqrt(children[0][i] / parent[i])

            s1nA = __SumDifferent(children[0], i)

            if (tn == 0.0):
                if s1nA == 0.0:
                    s1n = float('nan')
                else:
                    s1n = float('inf')
            else: s1n = math.sqrt(s1nA / tn)

            if parent[i] == 0.0:
                if children[1][i] == 0.0:
                    s2p = float('nan')
                else: s2p = float('inf')
            else: s2p = math.sqrt(children[1][i] / parent[i])

            s2nA = __SumDifferent(children[1], i)
            if (tn == 0.0):
                if s2nA == 0.0:
                    s2n = float('nan')
                else:
                    s2n = float('inf')
            else: s2n = math.sqrt(s2nA / tn)

            currentValue = math.pow(s1p - s1n, 2) + math.pow(s2p - s2n, 2)
            
            if currentValue > hellinger:
                hellinger = currentValue
    except ZeroDivisionError:
        raise Exception(f"Division by Zero ocurrend")

    if hellinger == float('inf'):
        return sys.float_info.max

    return math.sqrt(hellinger)

def QuinlanGain (parent, children):
    result = __GetImpurity(parent)
    total = sum(parent)
    nonMissing = 0.0

    for distribution in children:
        childCount = sum(distribution)
        nonMissing += childCount
        result -= __GetImpurity(distribution) * (childCount * 1.0 / total)

    return result * (nonMissing) / total 


def __SumDifferent(vector, index):
    sumValue = 0
    for i in range(len(vector)):
        if index != i:
            sumValue += vector[i]
    return sumValue

def __GetImpurity (distribution) :
    result = 0
    count = sum(distribution)

    for value in distribution:
        if (value != 0):
            p = value * 1.0 / count
            result -= p * math.log(p, 2)

    return result
