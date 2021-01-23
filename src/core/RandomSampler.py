import random

def rigSampler():
    return ['thal', 'slope', 'cp', 'oldpeak']

def SampleWithoutRepetition(poulation, sampleSize):

    result =  random.sample(list(
        map(lambda attribute: attribute, poulation)), sampleSize)
    return result