import os
import random
import math
import pandas as pd
import sys
from core.FileManipulation import ReadARFF, ReadDAT


class Dataset(object):

    def __init__(self, file):
        # region ARFF/DAT file handling
        arffFile = GetFromFile(file)

        # Check for class attribute before starting
        attributes = arffFile['attributes']
        if not list(filter(lambda attr: attr[0].strip().lower() == 'class', attributes)):
            raise Exception(
                f"This dataset does not contain an explicit class attribute!")

        self.Relation = arffFile['relation']

        self.Model = arffFile['attributes']

        self.Instances = pd.DataFrame.from_records(
            arffFile['data'], columns=list(
                map(lambda attribute: attribute[0], self.Model))).values

    @property
    def Attributes(self):
        return list(
            filter(lambda attr: attr[0].strip().lower() != 'class', self.Model))

    @property
    def Class(self):
        return list(
            filter(lambda attr: attr[0].strip().lower() == 'class', self.Model))[0]

    @property
    def AttributesInformation(self):
        return list(
            map(lambda attr: FeatureInformation(self, attr), self.Attributes))

    @property
    def ClassInformation(self):
        return FeatureInformation(self, self.Class)

    # region Attribute handling

    def GetAttribute(self, attribute):
        return list(filter(lambda attr: attr[0] == attribute, self.Model))[0]

    def GetAttributeNames(self):
        return list(
            map(lambda attribute: attribute[0], self.Model))

    def GetNominalValues(self, feature):
        attribute = list(
            filter(lambda attr: attr[0].lower() == feature.lower(), self.Model))

        if isinstance(attribute[0][1], list):
            return attribute[0][1]
        else:
            return None

    def GetValueOfIndex(self, feature, index):
        values = self.GetNominalValues(feature)
        if not values:
            return None
        else:
            return values[index]

    def GetIndexOfValue(self, feature, value):
        values = self.GetNominalValues(feature)
        if not values:
            return -1
        else:
            return values.index(value)

    def GetClasses(self):
        return self.GetNominalValues('class')

    def GetClassIdx(self):
        return self.GetFeatureIdx(self.Class)

    def GetFeatureIdx(self, feature):
        return self.Model.index(feature)

    def IsNominalFeature(self, feature):
        if isinstance(feature[1], str) and feature[1].lower() == 'string':
            raise Exception("String attributes are not supported!")
        return isinstance(feature[1], list)

    def GetFeatureValue(self, feature, instance):
        if isinstance(feature[1], list):
            return self.GetIndexOfValue(feature[0], instance[self.GetFeatureIdx(feature)])
        elif feature[1].lower() in ['numeric', 'real', 'integer']:
            return instance[self.GetFeatureIdx(feature)]
        else:
            raise Exception(
                "Attribute must be either nominal, numeric, real or integer")

    def IsMissing(self, feature, instance):
        value = self.GetFeatureValue(feature, instance)

        if self.IsNominalFeature(feature):
            return value < 0
        else:
            return (not value or value == math.nan)

    # endregion

    def ScalarProjection(self, instance, features, weights):

        if len(list(filter(lambda feature: self.IsMissing(feature, instance), features))) > 0:
            return math.nan

        result = sum([weights[feature] * self.GetFeatureValue(feature, instance)
                      for feature in features])

        return result


class FeatureInformation(object):
    def __init__(self, dataset, feature):
        self.Dataset = dataset
        self.Feature = feature
        self.MissingValueCount = 0
        self.MinValue = 0
        self.MaxValue = 0
        self.Distribution = []
        self.ValueProbability = []
        self.Ratio = []

        self.Initialize()

    def Initialize(self):

        featureIdx = self.Dataset.GetFeatureIdx(self.Feature)

        nonMissingValues = list(filter(lambda instance: not self.Dataset.IsMissing(
            self.Feature, instance), self.Dataset.Instances))

        self.MissingValueCount = len(
            self.Dataset.Instances) - len(nonMissingValues)

        if self.Dataset.IsNominalFeature(self.Feature):

            self.Distribution = [0]*len(self.Feature[1])
            for value in range(len(self.Distribution)):
                self.Distribution[value] = len(
                    list(filter(lambda instance: self.Dataset.GetFeatureValue(self.Feature, instance) == value and not self.Dataset.IsMissing(self.Feature, instance), self.Dataset.Instances)))
            self.ValueProbability = list(
                map(lambda value: value / sum(self.Distribution), self.Distribution))
            self.Ratio = list(
                map(lambda value: value / min(self.Distribution), self.Distribution))

        else:
            if len(nonMissingValues) > 0:
                self.MinValue = min(list(map(
                    lambda instance: instance[featureIdx], nonMissingValues)))
                self.MaxValue = max(list(map(
                    lambda instance: instance[featureIdx], nonMissingValues)))
            else:
                self.MinValue = 0
                self.MaxValue = 0


def GetFromFile(file):
    if os.path.isfile(file):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".arff":
            return ReadARFF(file)
        elif file_extension == ".dat":
            return ReadDAT(file)
        else:
            raise Exception(
                f"Extension '{file_extension}' of file '{filename}' is not supported ")
    else:
        raise Exception(f"File: {file} is not valid")
