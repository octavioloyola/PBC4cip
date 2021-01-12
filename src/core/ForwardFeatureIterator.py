from copy import copy, deepcopy


class ForwardFeatureIterator(object):
    def __init__(self, dataset, features):
        self.Dataset = dataset
        self.CandidateFeatures = features
        self.SelectedFeatures = list()

    def Add(self, feature):
        if feature not in self.CandidateFeatures:
            return False
        self.SelectedFeatures.append(feature)
        self.CandidateFeatures.remove(feature)
        return True

    @property
    def FeaturesRemain(self):
        return len(self.CandidateFeatures) > 0

    def GetFeatures(self):
        result = list()
        for feature in self.CandidateFeatures:
            features = copy(self.SelectedFeatures)
            features.append(feature)
            result.append(features)

        return result
