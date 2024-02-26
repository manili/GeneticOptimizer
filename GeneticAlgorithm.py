import random
from math import floor
from cmath import sin
from random import uniform
from typing import Tuple


def fooFunc(input: Tuple[float, float, float]) -> float:
    return 2 * input[0] + input[1] ** 3 + sin(input[2]) - 2.3673

def populationFunc(size: int) -> Tuple[float, float, float]:
    population = []
    for i in range(0, size):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        z = random.uniform(-100, 100)
        population.append( (x, y, z) )
    return population

def fitnessFunc(population: Tuple[float, float, float], precision: float, evaluationFunc: callable) -> tuple( (Tuple[float, float, float], float) ):
    fitness = []
    for thing in population:
        calc = evaluationFunc(thing)
        calcAbs = abs(calc)
        fitness.append( (thing, calcAbs) )
        if calcAbs <= precision:
            fitness.sort(key=lambda x: x[1])
            fitness.append( (thing, calc) )
            return fitness
    fitness.sort(key=lambda x: x[1])
    return fitness

def selectionMutationFunc(orderedPopulation: Tuple[float, float, float], selectionLen: int, populationLen: int) -> Tuple[float, float, float]:
    newGen = []
    for i in range(0, floor(populationLen / selectionLen)):
        for person in orderedPopulation:
            x = person[0][0] + random.uniform(-0.01, 0.01)
            y = person[0][1] + random.uniform(-0.01, 0.01)
            z = person[0][2] + random.uniform(-0.01, 0.01)
            newGen.append( (x, y, z) )
    return newGen

def printFitnessFunc(fitness: tuple( (Tuple[float, float, float], float) )):
    for item in fitness:
        print(item)

PRECISION = 0.0000001
population = populationFunc(1000)
fitness = fitnessFunc(population, PRECISION, fooFunc)
printFitnessFunc(fitness)

while fitness[0][1] > PRECISION:
    population = selectionMutationFunc(fitness[ : 100], 100, 1000)
    fitness = fitnessFunc(population, PRECISION, fooFunc)
    printFitnessFunc(fitness)

