import random

def SampleWithoutRepetition(poulation, sampleSize):
    random.seed(1000)
    #print(f"population: {poulation}")
    #print(f"sampleSize: {sampleSize}")
    result =  random.sample(set(
        map(lambda attribute: attribute, poulation)), sampleSize)
    print(f"resultt: {result}")
    return result