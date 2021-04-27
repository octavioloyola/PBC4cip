import random

def SampleWithoutRepetition(poulation, sampleSize):
    #random.seed(1)
    result =  random.sample(list(
        map(lambda attribute: attribute, poulation)), sampleSize)
    
    #fixedPopulation = ['a0', 'a1', 'a2', 'a3']
    #fixedPopulation = ['STDs(number)', 'Smokes', 'NumOfPregnancies', 'IUD']
    #fixedPopulation = ['V1', 'a1']
    #fixedPopulation = ['3', '5', '1', '8']
    fixedPopulation = ['sex', 'sick', 'tumor', 'lithium', 'goitre']
    return fixedPopulation
    #print(result)
    #return result