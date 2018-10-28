import random
from random import randint

#The initial settings:

varNum = 8                              #the number of variables / genes
clauseNum = 8                           #the number of clauses
initial = [0, 0, 0, 0, 0, 0, 0, 0]      #the initial values given to each variable
probability = 0.05                      #the probability of a mutation
rounds = 50                            #the number of rounds to be considered
printRounds = 1                         #print the results every printRounds rounds
orgNum = 16                            #the number of organisms

def clause0(values) :                   #definition of clause 0
    return values[0] or not values[2] or values[5]

def clause1(values) :                   #definition of clause 1
    return values[1] or values[4]

def clause2(values) :                   #definition of clause 2
    return values[2] or values[6]

def clause3(values) :                   #definition of clause 3
    return values[3] or values[1]

def clause4(values) :                   #definition of clause 4
    return values[4] or values[7]

def clause5(values) :                   #definition of clause 5
    return values[5] or values[1] or values[0]

def clause6(values) :                   #definition of clause 6
    return values[6] or values[7] or not values[4]

def clause7(values) :                   #definition of clause 7
    return values[7] or values[3] or values[2]

clauses = {clause0, clause1, clause2, clause3, clause4, clause5, clause6, clause7}

def getFitness(values) :                #get the fitness of an organism
    solvedClauses = 0
    for clause in clauses :
        solvedClauses += clause(values)
    return solvedClauses / clauseNum

def mutations(values) :                 #generate mutations for an organism; we consider each gene (variable) and mutate it with the initial probability set by the user
    for i in range(0, varNum) :
        p = random.random()
        if p < probability:
            values[i] = 1 - values[i]
    return values

def initialize(fitness, organisms):     #initialization of fitness and organisms
    for i in range (0, orgNum):
        fitness.append(0)
        organisms.append(initial)

def reproduce(fitness, organisms, org, fitAvg): #reproduce to 1 or 2 offsprings if fitness is above a threshold
    if fitness[org] > 0.3 :      
        organisms[randint(0, orgNum - 1)] = organisms[org]
        if fitness[org] > 0.5 :      
            organisms[randint(0, orgNum - 1)] = organisms[org]
            if fitness[org] > 0.75 :      
                organisms[randint(0, orgNum - 1)] = organisms[org]
                if fitness[org] > 0.85 :      
                    organisms[randint(0, orgNum - 1)] = organisms[org]
                    organisms[randint(0, orgNum - 1)] = organisms[org]
                    if fitness[org] > 0.95 :      
                        organisms[randint(0, orgNum - 1)] = organisms[org]
                        organisms[randint(0, orgNum - 1)] = organisms[org]

def printRes(fitness, organisms, round):
    print("Round: " + str(round) + "\n\n")
    for org in range (0, orgNum):
        print(str(org) + ": " + str(fitness[org]) + "   " + str(organisms[org]))

def avg(fitness):
    s = 0
    for x in fitness :
        s += x
    return s / orgNum

def simulation() :
    fitness = []
    organisms = []
    initialize(fitness, organisms)
    for round in range (0, rounds):
        if round % printRounds == 0 :
             print("Round: " + str(round) + "\n\n")
        for org in range (0, orgNum):
            mutations(organisms[org])                   #do the mutation
            fitness[org] = getFitness(organisms[org])   #compute the fitness
            if round % printRounds == 0 :
                print(str(org) + ": " + str(fitness[org]) + "   " + str(organisms[org]))
            reproduce(fitness, organisms, org, avg(fitness))          #reproduce
        if round % printRounds == 0 :
             print("Average fitness: " + str(avg(fitness)) + "\n")


simulation()


            