import random

def SampleWithoutRepetition(population, sample_size):
    print(f"population: {population}")
    result =  random.sample(list(
        map(lambda attribute: attribute, population)), sample_size)
    
    print(f"result: {result}")
    return result

def SampleAllList(population, sample_size):
    return population