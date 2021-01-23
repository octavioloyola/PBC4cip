import math
import random
import operator


def SumMatrix(matrix):
    if not matrix or len(matrix) == 0:
        return 0

    return sum(map(sum, matrix))


def ArgMin(source):
    if not source:
        return -1
    minValue = min(source)
    return source.index(minValue)


def ArgMax(source):
    if not source:
        return -1
    maxValue = max(source)
    return source.index(maxValue)


def MultiplyBy(array, value):
    return list(map(lambda element: element * value, array))


def AddTo(a, b):
    if len(a) != len(b):
        raise Exception
    return list(map(operator.add, a, b))


def Substract(a, b):
    if len(a) != len(b):
        raise Exception
    return list(map(operator.sub, a, b))


def CreateMembershipTupleList(instances):
    tupleList =  [(x,1.0) for x in instances]
    return tupleList

#Find Distribution of class values in dataset, i.e how many positive vs negative instances there are
def FindDistribution(source, model, classFeature):
    if isinstance(classFeature[1], str):
        raise Exception("Cannot find distribution for non-nominal class")

    result = [0]*len(classFeature[1])
    classIdx = model.index(classFeature)

    for element in source:
        value = classFeature[1].index(element[0][classIdx])
        if value >= 0:
            result[value] += element[1]
        else:
            continue

    return result

