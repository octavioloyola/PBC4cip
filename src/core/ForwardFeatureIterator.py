from copy import copy, deepcopy


class ForwardFeatureIterator(object):
    def __init__(self, dataset, features):
        self.__Dataset = dataset
        self.__CandidateFeatures = features
        self.__SelectedFeatures = list()
    
    @property
    def Dataset(self):
        return self.__Dataset
    @Dataset.setter
    def Dataset(self, new_dataset):
        self.__Dataset = new_dataset
    
    @property
    def CandidateFeatures(self):
        return self.__CandidateFeatures
    @CandidateFeatures.setter
    def CandidateFeatures(self, new_candidate_features):
        self.__CandidateFeatures = new_candidate_features

    @property
    def SelectedFeatures(self):
        return self.__Dataset
    @SelectedFeatures.setter
    def SelectedFeatures(self, new_selected_features):
        self.__SelectedFeatures = new_selected_features

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
            features.insert(0,feature)
            result.append(features)

        return result
