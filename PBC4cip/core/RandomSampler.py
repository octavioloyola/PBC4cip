import random

def SampleWithoutRepetition(population, sample_size):
    result =  random.sample(list(
        map(lambda attribute: attribute, population)), sample_size)
    
    return result

def SampleAllList(population, sample_size):
    return population