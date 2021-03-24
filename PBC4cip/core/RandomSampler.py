import random

def SampleWithoutRepetition(poulation, sampleSize):
    result =  random.sample(list(
        map(lambda attribute: attribute, poulation)), sampleSize)
    
    #fixedPopulation = ['a0', 'a1', 'a2', 'a3']
    fixedPopulation = ['a0', 'a2', 'a8', 'a4']
    #return fixedPopulation
    #print(result)
    return result