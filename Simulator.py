import random
from random import randint, shuffle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

class Clause:                    
    def __init__(self, elems, weight):
        self._elems = elems        #the elements in a clause
        self._weight = weight      #the weight for each clause
    def evaluate (self,variables):
        res = 1
        
        n = len(self._elems)
        
        for i in range(0,n):
            x = abs(self._elems[i]) - 1   # <0 => not; >0 => normal; we start from 1 / -1 (for clause 0)
            if self._elems[i] < 0 :
                res *= (1 - variables[x])
            else:
                res *= variables[x]
        return res * self._weight
    def getWeight(self):
        return self._weight


class Organism:
    def __init__(self, genome, env, prob):     #env = environment = list of clauses
        self._genome = genome
        self._env = env
        self._fitness = self._computeFitness()
        self._prob = prob
    
    def _computeFitness(self):
        solvedClauses = 0
        clauseNum = len(self._env)
        for clause in self._env :
            solvedClauses += clause.evaluate(self._genome)
        return solvedClauses + 1         #fitness >= 1
    
    def mutate(self):                        #triggers a mutation cycle where each gene can change with probability prob
        varNum = len(self._genome)
        for i in range(0, varNum) :
            doMutation = np.random.binomial(1, self._prob, 1)
            if doMutation == 1:
                self._genome[i] = 1 - self._genome[i]
        self._fitness = self._computeFitness()
        
    def offsprings(self, totalFitness, orgNum, label, dct):                    #returns a list of offsprings based on the current fitness
        offsprings = []
        myFitness = self._fitness               
        s = int(np.random.poisson(myFitness - 1, 1) + 1)   #draw a sample from Poisson distribution; we ensure that each organism has at least one offspring
        dct[label] = Organism(self._genome.copy(), self._env, self._prob);
        return (s, label)          #we return the number of offsprings and a label to identify their (common) genome
    
    def getFitness(self):
        return self._fitness
    
    def getGenome(self):
        return self._genome.copy()
    
    def computeMutants(self) :   #returns a list of all the possible mutants of an organism at Hamming distance 1
        res = []
        l = len(self._genome)
        for i in range (0, l) :
            newGenome = self._genome.copy()
            newGenome[i] = 1 - newGenome[i]
            res.append(Organism(newGenome, self._env, self._prob))
        return res


class LocalStatistics:     #computes local optimality statistics, regarding the fittest closest mutant
    def __init__(self, population, maxPossibleFitness) :
        self._population = population
        self._maxFitList = []      #fitness of the fittest mutant after each round
        self._avgMutList = []      #average fitness of the possible mutants per round
        self._avgFitterList = []   #average fitness of the possible mutants that have higher fitness per round
        self._maxPossibleFitness = maxPossibleFitness   #used to normalize fitness before plotting
        self._normList = []     #used to display the normalization equivalent
        self._neighbourData = {}
        self._max5SelList = []
        self._min5SelList = []
        self._maxSelList = []
        self._minSelList = []
        self._avgSelList = []
        
    def updateStats(self) :
        rnd = self._population.getRound()
        roundNeighbourData = {}
        (maxDFit, avgMut, avgFitter) = self._computeStats(roundNeighbourData)
        self._neighbourData[rnd] = roundNeighbourData
        self._maxFitList.append((rnd, maxDFit))
        self._avgMutList.append((rnd, avgMut))
        self._avgFitterList.append((rnd, avgFitter))
        self._normList.append((rnd, 1 / self._maxPossibleFitness))
        avgSel = self._computeSelection(roundNeighbourData)  #compute the selection coefficients statistics
        self._avgSelList.append((rnd, avgSel))
     
        
    def plotStats(self):     #plot all the stats so far; each stat can also be plotted separately
        plt.title("Local fitness statistics")
        plt.xlabel("Rounds")
        plt.ylabel(r'$\Delta$' + ' raw fitness')
        self.plotMaxFit()
        self.plotAvgMut()
        self.plotAvgFitter()
        self.plotNorm()
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        plt.title("Selection coefficient statistics")
        plt.xlabel("Rounds")
        plt.ylabel("Selection coefficient")
        self.plotAvgSel()
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
    def _fillPlot(self, extreme, middle):
        xs = [x[0] for x in extreme]  #fill between 2 lines
        ys1 = [x[1] for x in extreme]
        ys2 = [x[1] for x in middle]
        plt.fill_between(xs, ys1, ys2, color = "royalblue")
    
    def plotAvgSel(self):
        self._plotSelStat(self._avgSelList, "Average selection coefficient", "red")
    
    def _plotSelStat(self, statList, des, clr):     #plot the selection and fit the power law and exponential curves
        xs = [x[0] for x in statList]
        ys = [x[1] for x in statList]
        def expFunc(x, a, c, d):
            return a*np.exp(-c*x)+d
        popt, pcov = curve_fit(expFunc, xs, ys)
        plt.plot(xs,ys, 'ro', label = des, color = clr)
        yy = [expFunc(x, *popt) for x in xs]
        plt.plot(xs,yy, label = "Exponential decay", color = "blue")
        def powFunc(x, m, c, c0):
            return c0 + x**m * c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(powFunc, xs, ys)
            yy = [powFunc(x, *popt) for x in xs]
        plt.plot(xs,yy, label = "Power law decay", color = "yellow")
        
        
    def plotNorm(self):
        self._plotStat(self._normList, "Normalization value", '--')
        
    def plotMaxFit(self):
        self._plotStat(self._maxFitList, "Fittest possible mutant")
        
    def plotAvgMut(self):
        self._plotStat(self._avgMutList, "Average of possible mutants")
        
    def plotAvgFitter(self):
        self._plotStat(self._avgFitterList, "Average of possible mutants that are fitter")
        
    def _plotStat(self, statList, des, lineType = '-'):
        xs = [x[0] for x in statList]
        ys = [x[1] for x in statList]
        plt.plot(xs, ys, lineType, label = des)
        
    def _computeStats(self, roundNeighbourData):
        maxDFit = 0
        sAvgMut = 0
        nAvgMut = 0
        sAvgFitter = 0
        nAvgFitter = 0
        for org in self._population.getOrganisms():
            mutantsDeltaFitness = []
            mutants = org.computeMutants()
            for x in mutants:
                dFit = x.getFitness() - org.getFitness()
                mutantsDeltaFitness.append(dFit)
                if dFit > maxDFit:
                    maxDFit = dFit
                sAvgMut += dFit
                nAvgMut += 1
                if dFit > 0:    #fitter than its parent
                    sAvgFitter += dFit
                    nAvgFitter += 1
            roundNeighbourData[org] = (org.getFitness(), mutantsDeltaFitness)  #compute the neighbour data for for the current round
        if nAvgFitter == 0:
            nAvgFitter = 1
        return (maxDFit, sAvgMut / nAvgMut, sAvgFitter / nAvgFitter)
                                
    def _computeSelection(self, roundNeighbourData):
        selection = []
        for org in self._population.getOrganisms():
            (fit, mutDeltaFit) = roundNeighbourData[org]
            maxMutDeltaFit = max(mutDeltaFit)
            selection.append(maxMutDeltaFit / fit)     #compute the selection coefficient
        selection.sort()
        avgSel = np.mean(selection)
        return avgSel
        
                                
    def getNeighbourData(self):
        return self._neighbourData;    #this contains the local data (neighbour data) for a round, and organism, its fitness and its possible mutants fitness
    
    #For the visualisation of this:
    #Show data for a specific round / show data using joyplots (might not be easy to understand)
    #How to visualize the data for the population: 3D graph / joyplots / show separate distribution for organisms and then for mutants for each organism


class Statistics:                       #contains all the statistics and methods to update and plot them for a population
    def __init__(self, population, maxPossibleFitness, orgNum) :
        self._population = population
        self._avgFitList = []
        self._maxFitList = []
        self._minFitList = []
        self._max5PerFitList = []
        self._min5PerFitList = []
        self._maxPossibleFitness = maxPossibleFitness   #used to normalize fitness before plotting
        self._orgNum = orgNum
        
    def updateStats(self):
        rnd = self._population.getRound()
        self._avgFitList.append((rnd, self._computeAvgFit() / self._maxPossibleFitness))
        self._maxFitList.append((rnd, self._computeMaxFit() / self._maxPossibleFitness))
        self._minFitList.append((rnd, self._computeMinFit() / self._maxPossibleFitness))
        self._max5PerFitList.append((rnd, self._computeMax5PerFit() / self._maxPossibleFitness))
        self._min5PerFitList.append((rnd, self._computeMin5PerFit() / self._maxPossibleFitness))
        
        
    def plotStats(self):     #plot all the stats so far; each stat can also be plotted separately
        plt.title("General fitness statistics")
        plt.xlabel("Rounds")
        plt.ylabel("Normalized fitness")
        self.plotAvgFit()
        self.plotMinFit()
        self.plotMaxFit()
        self.plotMin5PerFit()
        self.plotMax5PerFit()
        self._fillPlot(self._maxFitList, self._max5PerFitList)  #fill between max and max 5 percent
        self._fillPlot(self._minFitList, self._min5PerFitList)  #fill between min and min 5 percent
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
    def _fillPlot(self, extreme, middle):
        xs = [x[0] for x in extreme]  #fill between 2 lines
        ys1 = [x[1] for x in extreme]
        ys2 = [x[1] for x in middle]
        plt.fill_between(xs, ys1, ys2, color = "royalblue")
        
    def plotMax5PerFit(self):
        self._plotStat(self._max5PerFitList, "Maximum 5% average", "green")
        
    def plotMin5PerFit(self):
        self._plotStat(self._min5PerFitList, "Minimum 5% average", "green")
    
    def plotAvgFit(self):
        self._plotStat(self._avgFitList, "Average", "red")
    
    def plotMaxFit(self):
        self._plotStat(self._maxFitList, "Maximum", "blue")
        
    def plotMinFit(self):
        self._plotStat(self._minFitList, "Minimum", "blue")
    
    def _plotStat(self, statList, des, clr):
        xs = [x[0] for x in statList]
        ys = [x[1] for x in statList]
        plt.plot(xs,ys, label = des, color = clr)
        
    def _computeAvgFit(self):
        s = 0
        orgs = self._population.getOrganisms()
        for i in range(0, self._orgNum):
            s += orgs[i].getFitness()
        return s / self._orgNum
        
    def _computeMaxFit(self):
        maxFit = 0
        orgs = self._population.getOrganisms()
        for i in range(0, self._orgNum):
            fit = orgs[i].getFitness()
            if fit > maxFit:
                maxFit = fit
        return maxFit
    
    def _computeMinFit(self):
        minFit = 1000000000
        orgs = self._population.getOrganisms()
        for i in range(0, self._orgNum):
            fit = orgs[i].getFitness()
            if fit < minFit:
                minFit = fit
        return minFit
    
    def _computeMin5PerFit(self):
        orgs = self._population.getOrganisms()
        fit = []
        for i in range(0, self._orgNum):
            fit.append(orgs[i].getFitness())
        fit.sort()
        l = int(math.ceil(len(fit) * 0.05))      #get only the first 5%
        fit = fit[:l]
        s = 0
        for x in fit:                            #compute the average of it
            s += x
        return s / l
    
    def _computeMax5PerFit(self):
        orgs = self._population.getOrganisms()
        fit = []
        for i in range(0, self._orgNum):
            fit.append(orgs[i].getFitness())
        fit.sort(reverse = True)
        l = int(math.ceil(len(fit) * 0.05))      #get only the first 5% (biggest 5%)
        fit = fit[:l]
        s = 0
        for x in fit:                            #compute the average of it
            s += x
        return s / l
    
    def plotGenome(self):
        orgs = self._population.getOrganisms()
        for org in orgs:
            print(str(org.getFitness()) + "   " + str(org.getGenome()))


class Population:                        #contains the current population and statistics; is updated on nextGeneration call
    def __init__(self, orgNum, env, prob, initial):
        self._orgNum = orgNum
        self._env = env
        self._prob = prob
        self._organisms = []
        self._round = 0
        self._maxPossibleFitness = 1    #used to normalize fitness when plotting
        
        for x in env:
            self._maxPossibleFitness += x.getWeight()

        self._stats = Statistics(self, self._maxPossibleFitness, self._orgNum)
        self._localStats = LocalStatistics(self, self._maxPossibleFitness)    

        for i in range(0, self._orgNum):
            self._organisms.append(Organism(initial, self._env, self._prob))
        self._stats.updateStats()
        self._localStats.updateStats()

    def getOrganisms(self) :
        return self._organisms.copy()
    
    def getRound(self) :
        return self._round
    
    def getStats(self) :
        return self._stats
    
    def getLocalStats(self) :
        return self._localStats
    
    def nextGeneration(self):
        nextGenPool = []
        self._round += 1
        totalFitness = 0                                        #compute total fitness for offspring generation
        for i in range(0, self._orgNum):
            self._organisms[i].mutate()                          #generate mutations
            totalFitness += self._organisms[i].getFitness()
        dct = {}                                                 #map a label to an organism; use it only after getting n from the next generation pool
        for i in range(0, self._orgNum):
            nextGenPool.append(self. _organisms[i].offsprings(totalFitness, self._orgNum, i, dct))       #compute the next generation pool (of pairs (number of orgs, label) )
        
        sumNextPool = 0
        
        for (num, label) in nextGenPool:                       #compute the total number of orgs in next gen pool
        	sumNextPool += num
        totalRemaining = self._orgNum                                #remaining number of orgs to complete
        self._organisms = []
        
        for (num, label) in nextGenPool[:-1]:
            if totalRemaining <= 0:
                currNum = 0
            else:
                currNum = np.random.hypergeometric(num, sumNextPool - num, totalRemaining)     #sample from the hypergeometric distribution the current number of organisms to add
            sumNextPool -= num
            totalRemaining -= currNum
            for i in range (0, currNum):
                self._organisms.append(dct[label])
       
        for i in range (0, totalRemaining):                      #add the remaining ones
        	self._organisms.append(dct[label])
  
        self._stats.updateStats()                                #update the stats for the new generation
        self._localStats.updateStats()


class Simulator:  
    def __init__(self, 
                initial,         #the initial values given to each variable / initial genome
                probability,     #the probability of a mutation
                rounds,          #the number of rounds to be considered
                orgNum,          #the number of organisms
                clauses):        #the list of clauses
        self._initial = initial
        self._probability = probability
        self._rounds = rounds
        self._orgNum = orgNum
        self._clauses = clauses
        self._population = Population(self._orgNum, self._clauses, self._probability, self._initial)
    
    def run(self):
        for i in range (0, self._rounds):
            self._population.nextGeneration()    
    def printLocalStatistics(self):
    	self._population.getLocalStats().plotStats()
    	
    def printStatistics(self):
    	self._population.getStats().plotStats()
