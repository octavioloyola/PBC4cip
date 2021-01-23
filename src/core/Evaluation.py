import math
import copy
from core.Helpers import ArgMax


class BasicEvaluation(object):
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.TPrate = 0
        self.TNrate = 0
        self.FPrate = 0
        self.FNrate = 0

        self.specificity = 0
        self.sensitivity = 0
        self.precision = 0
        self.recall = 0
        self.Yrate = 0


class ConfusionMatrix(object):
    def __init__(self, classes):
        numclass = len(classes)
        self.Classes = classes
        self.__lettersposition = list(
            map(lambda x: str(chr(x+ord('a'))), list(range(26))))
        self.Matrix = [[0 for i in range(numclass)] for j in range(numclass+1)]

    def ComputeBasicEvaluation(self, positiveClass):
        basicEvaluation = BasicEvaluation()
        N = sum(map(sum, self.Matrix))
        basicEvaluation.TP = self.Matrix[positiveClass][positiveClass]

        for i in range(len(self.Matrix[0])):
            basicEvaluation.TN += self.Matrix[i][i]
            basicEvaluation.FP += self.Matrix[positiveClass][i]

        basicEvaluation.TN -= basicEvaluation.TP
        basicEvaluation.FP -= basicEvaluation.TP

        # to add abstentions
        basicEvaluation.FP += self.Matrix[len(self.Matrix) - 1][positiveClass]

        basicEvaluation.FN = N - \
            (basicEvaluation.TP + basicEvaluation.FP + basicEvaluation.TN)

        try:
            basicEvaluation.TPrate = basicEvaluation.TP * \
                1.0 / (basicEvaluation.TP + basicEvaluation.FN)
        except ZeroDivisionError:
            basicEvaluation.TPrate = 0.0
        try:
            basicEvaluation.TNrate = basicEvaluation.TN * \
                1.0 / (basicEvaluation.TN + basicEvaluation.FP)
        except ZeroDivisionError:
            basicEvaluation.TNrate = 0.0
        try:
            basicEvaluation.FPrate = basicEvaluation.FP * \
                1.0 / (basicEvaluation.TN + basicEvaluation.FP)
        except ZeroDivisionError:
            basicEvaluation.FPrate = 0.0
        try:
            basicEvaluation.FNrate = basicEvaluation.FN * \
                1.0 / (basicEvaluation.TP + basicEvaluation.FN)
        except ZeroDivisionError:
            basicEvaluation.FNrate = 0.0

        basicEvaluation.sensitivity = basicEvaluation.TPrate
        basicEvaluation.specificity = basicEvaluation.TNrate

        try:
            basicEvaluation.precision = basicEvaluation.TP * \
                1.0 / (basicEvaluation.TP + basicEvaluation.FP)
        except ZeroDivisionError:
            basicEvaluation.precision = 0

        basicEvaluation.recall = basicEvaluation.TPrate

        try:
            basicEvaluation.Yrate = (
                basicEvaluation.TP + basicEvaluation.FP) * 1.0 / N
        except ZeroDivisionError:
            basicEvaluation.Yrate = 0

        return basicEvaluation

    def AUCMeasure(self, positiveClass):
        basiceval = self.ComputeBasicEvaluation(positiveClass)
        return (1 + (basiceval.TPrate - basiceval.FPrate))/2

    def __repr__(self):
        lettersClasses = '|\t'.join(
            [self.__lettersposition[i] for i in range(len(self.Classes))])
        headers_row = '|\t'.join(
            [":--:" for i in range(len(self.Classes))])
        result = f"|\t{lettersClasses}|\t<-- classified as |\r\n|\t{headers_row}|\t---|\r\n"
        for row in range(len(self.Matrix)):
            result += "|\t"+'|\t'.join([str(self.Matrix[row][col])
                                        for col in range(len(self.Matrix[0]))])
            if row < len(self.Matrix)-1:
                result += f"\t| {self.__lettersposition[row]} = {self.Classes[row]}|\r\n"
            if row == len(self.Matrix)-1:
                result += f"\t| The last row correspond to the abstentions|\r\n"

        return result


class CrispAndPartitionEvaluation(object):
    def __init__(self):
        self.ConfusionMatrix = None


def Evaluate(classes, real, predicted):
    if len(real) != len(predicted):
        raise Exception(
            "Cannot evaluate classification. Real and Predicted counts are different.")
    numClasses = len(classes)
    evaluation = CrispAndPartitionEvaluation()
    evaluation.ConfusionMatrix = ConfusionMatrix(classes)
    confusion = [[0]*2 for i in enumerate(classes)]
    classified_as = 0
    error_count = 0

    for i in range(len(real)):
        if real[i] != predicted[i]:
            error_count = error_count + 1
        confusion[real[i]][predicted[i]] = confusion[real[i]][predicted[i]] + 1

    acc = 100.0 * (len(real) - error_count) / len(real)
    auc = __obtainAUCMulticlass(confusion, len(classes))
        
    
    return confusion, acc, auc

def AddMatrices(cmA, cmB):
    if not cmA:
        return cmB
    if not cmB:
        return cmA

    if len(cmA) != len(cmB):
        raise Exception("Matrix missmatch")

    newMatrix = ConfusionMatrix(cmA.Classes)

    newMatrix.Matrix = [[cmA[i][j] + cmB[i][j]
                         for j in range(len(cmA[0]))] for i in range(len(cmA))]
    return newMatrix


def NormalizeVotes(values):
    result = [0]*len(values)
    argMax = ArgMax(values)

    if values[argMax] == 0:
        return result

    result[argMax] = 1
    return result

def __obtainAUCBinary(tp, tn, fp, fn):
        nPos = tp +fn
        nNeg = tn + fp
        recall = tp / nPos
        if tp == 0:
            recall = 0
        
        sensibility = tn/ nNeg
        if tn == 0:
            sensibility = 0
        
        return (recall + sensibility) / 2

def __obtainAUCMulticlass(confusion, num_classes):
    sumVal = 0
    for i in range(num_classes):
        tp = confusion[i][i]

        for j in range(i+1, num_classes):
            fp = confusion[j][i]
            fn = confusion[i][j]
            tn = confusion[j][j]
            sumVal = sumVal + __obtainAUCBinary(tp, tn, fp, fn)
    
    avg = (sumVal * 2) / (num_classes * (num_classes-1))
    return avg

    

