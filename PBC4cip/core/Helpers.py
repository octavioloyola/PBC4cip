import math
import random
import operator
import numpy as np
import pandas as pd
from itertools import chain

def SumMatrix(matrix):
    if not matrix or len(matrix) == 0:
        return 0

    return sum(map(sum, matrix))

def smallest_idx(source):
    if len(source) == 0:
        raise Exception("source must have at least 1 element")
    val = source[0]
    idx = 0
    for i,value in enumerate(source):
        if value < val:
            idx = i
            val = value 
    return idx

def random_small_idx(source, random_size):
    if len(source) == 0:
        raise Exception("source must have at least 1 element")
    
    if len(source) <= random_size:
        return random.randint(0, len(source)-1)

    lst = np.array(source)
    idx = np.argpartition(lst, random_size)
    small_idx = idx[0:random_size]
    r = random.randint(0, random_size -1)
    return small_idx[r]

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

def combine_instances(X, y):
    combined_list = []
    for i,val in enumerate(X):
        combined_list.append(__chain_together(val, y[i]))
    result = np.asarray(combined_list, dtype= np.object)
    return result

def __chain_together(a, b):
    return list(chain(*[a,b]))

#Find Distribution of class values in dataset, i.e how many positive vs negative instances there are
def FindDistribution(source, model, classFeature):
    if not isinstance(classFeature[1], list):
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

def convert_to_ndarray(y):
    new_y = []
    for y_elem in y:
         new_y.append(np.array([y_elem], dtype=np.object))
    pd_series = pd.Series(new_y, name='class')
    return pd_series

def get_col_dist(source):
    elems = set()
    for elem in source:
        if isinstance(elem, float):
            if not math.isnan(elem):
                elems.add(elem)
        else:
            elems.add(elem)

    return sorted(elems)

def get_idx_val(source, instance):
    for idx, val in enumerate(source):
        if instance == val:
            return idx
    raise Exception(f"Nominal value inside training dataset not found inside testing dataset")

