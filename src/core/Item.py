from core.FeatureSelectors import SingleFeatureSelector, CutPointSelector, MultipleValuesSelector, ValueAndComplementSelector, MultivariateCutPointSelector
from core.Dataset import Dataset
import math


class SubsetRelation(object):
    Unknown = 0
    Unrelated = 1
    Equal = 2
    Subset = 3
    Superset = 4
    Different = 5


class Item(object):

    def __init__(self, dataset, feature):
        self.Dataset = dataset
        self.Feature = feature

    def IsMatch(self, instance):
        return False

    def CompareTo(self, other):
        return SubsetRelation.Unknown

# region Univariate items


class SingleValueItem(Item):

    def __init__(self, dataset, feature, value):
        super().__init__(dataset, feature)
        self.Value = value

    def IsMatch(self, instance):
        return not self.Dataset.IsMissing(self.Feature, instance)

    def GetValue(self, instance):
        return self.Dataset.GetFeatureValue(self.Feature, instance)

    def GetValueRepresentation(self):
        if self.Dataset.IsNominalFeature(self.Feature):
            return self.Feature[1][self.Value]
        else:
            return self.Value

    def __repr__(self):
        return f"{self.Feature[0]} ? {self.GetValueRepresentation()}"


class EqualThanItem(SingleValueItem):

    def IsMatch(self, instance):
        if super().IsMatch(instance):
            return self.GetValue(instance) == self.Value
        return False

    def CompareTo(self, other):
        if isinstance(other, EqualThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Equal
            else:
                return SubsetRelation.Unrelated

        if isinstance(other, DifferentThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Unrelated
            if self.Dataset.IsNominalFeature(self.Feature):
                numberOfValues = len(self.Feature[1])
                if self.Value != other.Value:
                    if numberOfValues == 2:
                        return SubsetRelation.Equal
                    else:
                        return SubsetRelation.Subset

        return SubsetRelation.Unrelated

    def __repr__(self):
        return f"{self.Feature[0]} = {self.GetValueRepresentation()}"


class DifferentThanItem(SingleValueItem):

    def IsMatch(self, instance):
        if super().IsMatch(instance):
            return self.GetValue(instance) != self.Value
        return False

    def CompareTo(self, other):
        if isinstance(other, DifferentThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Equal
            else:
                return SubsetRelation.Unrelated

        if isinstance(other, EqualThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Unrelated
            if self.Dataset.IsNominalFeature(self.Feature):
                numberOfValues = len(self.Feature[1])
                if self.Value != other.Value:
                    if numberOfValues == 2:
                        return SubsetRelation.Equal
                    else:
                        return SubsetRelation.Superset

        return SubsetRelation.Unrelated

    def __repr__(self):
        return f"{self.Feature[0]} != {self.GetValueRepresentation()}"


class LessOrEqualThanItem(SingleValueItem):

    def IsMatch(self, instance):
        if super().IsMatch(instance):
            return self.GetValue(instance) <= self.Value
        return False

    def CompareTo(self, other):
        if isinstance(other, LessOrEqualThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Equal
            if self.Value > other.Value:
                return SubsetRelation.Superset
            else:
                return SubsetRelation.Subset

        return SubsetRelation.Unrelated

    def __repr__(self):
        return f"{self.Feature[0]} <= {self.GetValueRepresentation():.3f}"


class GreatherThanItem(SingleValueItem):

    def IsMatch(self, instance):
        if super().IsMatch(instance):
            return self.GetValue(instance) > self.Value
        return False

    def CompareTo(self, other):
        if isinstance(other, GreatherThanItem):
            if self.Value == other.Value:
                return SubsetRelation.Equal
            if self.Value > other.Value:
                return SubsetRelation.Subset
            else:
                return SubsetRelation.Superset

        return SubsetRelation.Unrelated

    def __repr__(self):
        return f"{self.Feature[0]} > {self.GetValueRepresentation():.3f}"


class ItemBuilder(object):
    def GetItem(self, generalSelector, index):
        return None


class CutPointBasedBuilder(ItemBuilder):
    def GetItem(self, generalSelector, index):
        if not isinstance(generalSelector, CutPointSelector):
            raise Exception(
                f"Unexpected type of selector {generalSelector.__class__.__name__}. Was expecting CutPointSelector")

        if index == 0:
            return LessOrEqualThanItem(generalSelector.Dataset, generalSelector.Feature, generalSelector.CutPoint)
        elif index == 1:
            return GreatherThanItem(generalSelector.Dataset, generalSelector.Feature, generalSelector.CutPoint)
        else:
            raise Exception("Invalid index value for CutPointSelector")


class ValueAndComplementBasedBuilder(ItemBuilder):
    def GetItem(self, generalSelector, index):
        if not isinstance(generalSelector, ValueAndComplementSelector):
            raise Exception(
                f"Unexpected type of selector {generalSelector.__class__.__name__}. Was expecting ValueAndComplementSelector")
        if index == 0:
            return EqualThanItem(generalSelector.Dataset, generalSelector.Feature, generalSelector.Value)
        elif index == 1:
            return DifferentThanItem(generalSelector.Dataset, generalSelector.Feature, generalSelector.Value)
        else:
            raise Exception(
                "Invalid index value for ValueAndComplementSelector")


class MultipleValuesBasedBuilder(ItemBuilder):
    def GetItem(self, generalSelector, index):
        if not isinstance(generalSelector, MultipleValuesSelector):
            raise Exception(
                f"Unexpected type of selector {generalSelector.__class__.__name__}. Was expecting ValueAndComplementSelector")
        if index < 0 or index >= len(generalSelector.Values):
            raise Exception(
                "Invalid index value for MultipleValuesSelector")
        return EqualThanItem(generalSelector.Dataset, generalSelector.Feature, generalSelector.Values[index])
# endregion

# region Multivariate items


class MultivariateSingleValueItem(Item):
    def __init__(self, dataset, features, value, weights):
        super().__init__(dataset, None)
        self.Value = value
        self.Weights = weights
        self.Features = features
        self.FeaturesHash = sum([hash(feature[0])
                                 for feature in self.Features])
        self._parallel = 0.001


class MultivariateLessOrEqualThanItem(MultivariateSingleValueItem):

    def IsMatch(self, instance):
        instanceValue = self.Dataset.ScalarProjection(
            instance, self.Features, self.Weights)
        if math.isnan(instanceValue):
            return False
        return instanceValue <= self.Value

    def CompareTo(self, other):
        if isinstance(other, MultivariateLessOrEqualThanItem):
            if self.FeaturesHash != other.FeaturesHash:
                return SubsetRelation.Unrelated
            if len(self.Features) != len(other.Features):
                return SubsetRelation.Unrelated

            try:
                proportion = list(other.Weights.values())[
                    0] / list(self.Weights.values())[0]
                for feature in list(self.Weights.keys()):
                    if abs(self.Weights[feature]*proportion - other.Weights[feature]) > self._parallel:
                        return SubsetRelation.Unrelated

                if abs(self.Value * proportion - other.Value) < self._parallel:
                    return SubsetRelation.Equal

                if self.Value * proportion > other.Value:
                    return SubsetRelation.Superset
                else:
                    return SubsetRelation.Subset
            except ZeroDivisionError:
                return SubsetRelation.Unrelated

        return SubsetRelation.Unrelated

    def __repr__(self):
        linearCombination = ' + '.join(
            map(lambda weight: str(self.Weights[weight]) + " * " + weight[0], self.Weights))
        return f"{linearCombination} <= {self.Value}"


class MultivariateGreatherThanItem(MultivariateSingleValueItem):

    def IsMatch(self, instance):
        instanceValue = self.Dataset.ScalarProjection(
            instance, self.Features, self.Weights)
        if math.isnan(instanceValue):
            return False
        return instanceValue <= self.Value

    def CompareTo(self, other):
        if isinstance(other, MultivariateGreatherThanItem):
            if self.FeaturesHash != other.FeaturesHash:
                return SubsetRelation.Unrelated
            if len(self.Features) != len(other.Features):
                return SubsetRelation.Unrelated

            try:
                proportion = list(other.Weights.values())[
                    0] / list(self.Weights.values())[0]
                for feature in list(self.Weights.keys()):
                    if abs(self.Weights[feature]*proportion - other.Weights[feature]) > self._parallel:
                        return SubsetRelation.Unrelated

                if abs(self.Value * proportion - other.Value) < self._parallel:
                    return SubsetRelation.Equal

                if self.Value * proportion > other.Value:
                    return SubsetRelation.Subset
                else:
                    return SubsetRelation.Superset
            except ZeroDivisionError:
                return SubsetRelation.Unrelated

        return SubsetRelation.Unrelated

    def __repr__(self):
        linearCombination = ' + '.join(
            map(lambda weight: str(self.Weights[weight]) + " * " + weight[0], self.Weights))

        return f"{linearCombination} <= {self.Value}"


class MultivariateCutPointBasedBuilder(ItemBuilder):
    def GetItem(self, generalSelector, index):
        if not isinstance(generalSelector, MultivariateCutPointSelector):
            raise Exception(
                f"Unexpected type of selector {generalSelector.__class__.__name__}. Was expecting CutPointSelector")
        if index == 0:
            return MultivariateLessOrEqualThanItem(
                generalSelector.Dataset, generalSelector.Features, generalSelector.CutPoint, generalSelector.Weights)
        elif index == 1:
            return MultivariateGreatherThanItem(
                generalSelector.Dataset, generalSelector.Features, generalSelector.CutPoint, generalSelector.Weights)
        else:
            raise Exception("Invalid index value for CutPointSelector")


# endregion


class ItemComparer(object):
    def Compare(self, left, right):
        if left.Feature == right.Feature or (not left.Feature and not right.Feature):
            return left.CompareTo(right)
        return SubsetRelation.Unrelated
