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

def Twoing(parent, children):
    if len(children) != 2:
        raise Exception("Twoing needs only 2 child nodes")

    if sum(parent) == max(parent):
        return 0
    
    total = sum(parent)
    SL = sum(children[0])
    SR = sum(children[1])
    twoing = 0.25 * SL/total * SR/total

    aux = 0
    for i,elems in enumerate(parent):
        aux = aux + abs((children[0][i] / SL) - (children[1][i]/ SR))

    twoing = twoing * (aux**2)
    return twoing

def GiniImpurity(parent, children):
    result = __GetImpurityGini(parent)
    total = sum(parent)
    nonMissing = 0

    for distribution in children:
        child_count = sum(distribution)
        nonMissing = nonMissing + child_count
        result = result - __GetImpurityGini(distribution) * (child_count * 1.0/total)
    
    return result * (nonMissing) / total

def ChiSquared(parent, children):
    #print(f"parent: {parent} children: {children}")
    if len(children) != 2:
        raise Exception(
            "Chi-Squared needs only two child nodes (binary split)")

    if sum(parent) == max(parent):
        return 0
    try:
        s1p = children[0][0] / parent[0]
        s1n = children[0][1] / parent[1]

        s2p = children[1][0] / parent[0]
        s2n = children[1][1] / parent[1]
    
        result = (math.pow(s1p-s1n, 2) / (s1p+s1n)) + (math.pow(s2p-s2n, 2) / (s2p+s2n))
        #print(f"return: {result}")
        return result
    except ZeroDivisionError:
        #print(f"return: nan")
        return float('nan')

def DKM (parent, children):
    result = __G(parent)
    total = sum(parent)
    non_missing = 0

    for distribution in children:
        child_count = sum(distribution)
        non_missing += child_count
        result -= __G(distribution) * (child_count * 1.0 / total)
    return result * (non_missing) / total

def G_Statistic(parent, children):
    result = __GetImpurity(parent)
    total = sum(parent)
    non_missing = 0

    for distribution in children:
        child_count = sum(distribution)
        non_missing += child_count
        result -= __GetImpurity(distribution) * (child_count * 1.0 / total)

    return 2 * sum(parent)*result * (non_missing) / total

def MARSH(parent, children):
    k = len(children)
    result = __GetImpurity(parent)

    total = sum(parent)
    non_missing = 0
    correction = 1

    for distribution in children:
        child_count = sum(distribution)
        non_missing += child_count
        result -= __GetImpurity(distribution) * (child_count * 1.0 / total)
        correction *= child_count / total
    
    return correction * math.pow(k,k) * result * (non_missing) / total

def KolmogorovDependence(parent, children):
    if len(children) != 2:
        raise Exception("Kolmogorov needs only 2 child nodes")

    kolmogorv = float('-inf')
    for i,value in enumerate(parent):
        try:
            F0 = children[0][i] / parent[i]
        except ZeroDivisionError:
            F0 = float('nan')
        try:
            F1 = __SumDifferent(children[0], i) / __SumDifferent(parent, i)
        except ZeroDivisionError:
            F1 = float('nan')
        curr_value = abs(F0-F1)

        if curr_value > kolmogorv:
            kolmogorv = curr_value
        if len(parent) == 2:
            break
    
    if kolmogorv == float('inf'):
        return sys.float_info.max
    
    return kolmogorv

def NormalizedGain(parent, children):
    result = __GetImpurity(parent)
    total = sum(parent)
    non_missing = 0

    for distribution in children:
        child_count = sum(distribution)
        non_missing += child_count
        result -= __GetImpurity(distribution) * (child_count * 1.0 / total)
    
    result /= math.log(len(children), 2)
    return result

def MultiClassBhattacharyya(parent, children):
    if len(children) != 2:
        raise Exception(f"Multi class Bhattacharyya needs only 2 nodes \
        for its children ")
    if sum(parent) == max(parent):
        return 0
    bhattacharyya = float('-inf')
    for i,value in enumerate(parent):
        try:
            negativeTotal =  __SumDifferent(parent, i)
            positiveLeft = math.sqrt(children[0][i] / parent[i])
            negativeLeft = math.sqrt(__SumDifferent(children[0], i) / negativeTotal)
            positiveRight = math.sqrt(children[1][i] / parent[i])
            negativeRight = math.sqrt(__SumDifferent(children[1],i) / negativeTotal)
            curr_value = math.sqrt(1-(math.sqrt(positiveLeft * negativeLeft) + math.sqrt(positiveRight * negativeRight)))
        except ValueError:
            curr_value = float('nan')
        except ZeroDivisionError:
            curr_value = float('nan')
        if curr_value > bhattacharyya:
            bhattacharyya = curr_value
        if bhattacharyya == float('inf'):
            raise Exception('Infinite value in Bhattacharyya')
        try:
            res = math.sqrt(bhattacharyya)
        except ValueError:
            res = float('nan')
    return res
        
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

def __GetImpurityGini (distribution) :
    result = 1.0
    count = sum(distribution)

    for value in distribution:
        if (value != 0):
            p = value * 1.0 / count
            result -= math.pow(p, 2)

    return result

def __G(distribution):
    result = 0
    count = sum(distribution)

    for value in distribution:
        if value != 0:
            p = value * 1.0/count
            result = result + 2* math.sqrt(p * (1-p))

    return result