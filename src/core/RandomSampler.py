import random

def SampleWithoutRepetition(poulation, sampleSize):
    #random.seed(100)
    print(f"population: {poulation}")
    #print(f"sampleSize: {sampleSize}")
    #Cleveland
    #fixedPopulation = ['thal', 'slope', 'cp', 'oldpeak']
    #fixedPopulation = ['oldpeak', 'slope', 'cp', 'thal']
    #fixedPopulation = ['age', 'cp', 'exang', 'chol']
    #fixedPopulation = ['trestbps', 'fbs', 'oldpeak', 'thalach']

    #Iris
    #fixedPopulation = ['sepalwidth', 'petallength', 'petalwidth']

    #Car
    fixedPopulation = ['Maint', 'Persons', 'Safety', 'Lug_boot']


    result =  random.sample(list(
        map(lambda attribute: attribute, poulation)), sampleSize)
    print(f"resultt: {result}")
    #return fixedPopulation
    return result