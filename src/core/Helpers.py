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


def CreateMembershipTuple(instances):
    tuples = tuple(map(lambda instance: (instance, 1.0), instances))
    return tuples


def FindDistribution(source, model, classFeature):
    #print(f"classFeature: {classFeature}")

    if isinstance(classFeature[1], str):
        raise Exception("Cannot find distribution for non-nominal class")

    result = [0]*len(classFeature[1])
    classIdx = model.index(classFeature)
    #print(f"model: {model}")
    #print(f"classIdx: {classIdx}")

    for element in source:
        value = classFeature[1].index(element[0][classIdx])
        if value >= 0:
            result[value] += element[1]
        else:
            continue

    #print(f"result in FindDistribution: {result}")
    return result
