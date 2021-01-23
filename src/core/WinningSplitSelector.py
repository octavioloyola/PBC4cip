import sys
import math
import copy
from _functools import cmp_to_key


class WinningSplitSelector:

    def __init__(self, whichBetterToFind=None):
        if not whichBetterToFind or whichBetterToFind <= 0:
            raise Exception("WhichBetterToFind must be positive")
        else:
            self.__whichBetterToFind = whichBetterToFind
        self.MinStoredValue = sys.float_info.min
        self.List = list(tuple())

    @property
    def WinningSelector(self):
        index = min(self.__whichBetterToFind - 1, len(self.List) - 1)
        return self.List[index][1]

    @property
    def WinningDistribution(self):
        index = min(self.__whichBetterToFind - 1, len(self.List) - 1)
        return self.List[index][2]

    def EvaluateThis(self, currentGain, splitIterator, level):

        if (len(self.List) < self.__whichBetterToFind or currentGain > self.MinStoredValue):
            currentChildSelector = splitIterator.CreateCurrentChildSelector()
            copyOfCurrentDistribution = copy.deepcopy(
                splitIterator.CurrentDistribution)
            self.List.append(
                tuple((currentGain, currentChildSelector, copyOfCurrentDistribution)))
            self.List.sort(key=cmp_to_key(Compare))

            if len(self.List) > self.__whichBetterToFind:
                self.List.remove(self.List[self.__whichBetterToFind])
            index = min(self.__whichBetterToFind-1, len(self.List)-1)

            self.MinStoredValue = self.List[index][0]
            return True
        return False

    def IsWinner(self):
        return len(self.List) > 0


def Compare(x, y):
    val = x[0]-y[0]
    if val > 0:
        return -1
    elif val < 0:
        return 1
    else:
        return 0
