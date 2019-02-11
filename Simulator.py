import random
from random import randint, shuffle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
from abc import ABCMeta, abstractmethod
import networkx as nx
from nxpd import draw

class Constraint:
	_metaclass_ = ABCMeta
	
	@abstractmethod
	def evaluate (self,variables): pass
	
	@abstractmethod
	def getWeight(self): pass

class ConstraintSat(Constraint):     #sat-clause constraint                    
    def __init__(self, elems, weight):
        self._elems = elems        #the elements in a clause
        self._weight = weight      #the weight for each clause
    def evaluate (self, variables, domains):  #for this type of constraint, we can ignore the domains, as it must be binary
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
        
class ConstraintBinaryModelUnary(Constraint):         #unary constraint for the binary model
    def __init__(self, elem, weight):
        self._elem = elem        	#the variable / gene to which it refers
        self._weight = weight       #the weight attributed to it
        
    def evaluate(self, variables, domains):   #for this type of constraint, we can ignore the domains, as it must be binary
        if variables[self._elem] == 1 :
            return self._weight
        return 0
    def getWeight(self):
        return max(0,self._weight)
    def type(self):
    	return "21"
    def getElem(self):
    	return self._elem
    def getWeights(self):
        return self._weight
    
class ConstraintBinaryModelBinaryDifferent(Constraint):         #binary constraint for the binary model, the first type
    def __init__(self, elems, weights):
        self._elems = elems        	  #the variables / genes to which it refers (an array of 2 elements)
        self._weights = weights       #the weights attributed to it (an array of 2 elements)
        
    def evaluate(self, variables, domains):   #for this type of constraint, we can ignore the domains, as it must be binary
        if variables[self._elems[0]] == 0 and variables[self._elems[1]] == 1 :
            return self._weights[0]
        if variables[self._elems[0]] == 1 and variables[self._elems[1]] == 0 :
            return self._weights[1]
        return 0
    def getWeight(self):
        return max(0,np.amax(self._weights))
    def type(self):
    	return "221"
    def getElems(self):
    	return self._elems
    def getWeights(self):
        return self._weights
        
class ConstraintBinaryModelBinarySame(Constraint):         #binary constraint for the binary model, the second type
    def __init__(self, elems, weights):
        self._elems = elems        	  #the variables / genes to which it refers (an array of 2 elements)
        self._weights = weights       #the weights attributed to it (an array of 2 elements)
        
    def evaluate(self, variables, domains):   #for this type of constraint, we can ignore the domains, as it must be binary
        if variables[self._elems[0]] == 0 and variables[self._elems[1]] == 0 :
            return self._weights[0]
        if variables[self._elems[0]] == 1 and variables[self._elems[1]] == 1 :
            return self._weights[1]
        return 0
    def getWeight(self):
        return max(0,np.amax(self._weights))
    def type(self):
        return "222"
    def getElems(self):
        return self._elems
    def getWeights(self):
        return self._weights
        
class ConstraintVCSP(Constraint):					 #the most general constraint
	def __init__(self, elems, weightInfo):   	 #we are given elements and the weightInfo (a weight function and a maximum weight value that is used in the statistics)
		self._elems = elems 					 #the variables / genes to which it refers (an array)
		self._weightFunction = weightInfo[0]      #get the weight function
		self._maxWeight = weightInfo[1]          #get the max weight
	def evaluate(self, variables, domains):
		args = []
		for elem in self._elems:
			args.append(domains[elem][variables[elem]])
		return self._weightFunction(args)
	def getWeight(self):
		return self._maxWeight

class ConstraintWCSP(Constraint):					#the general wcsp
    def __init__(self, elems, weight):
        self._vars = elems[0]					#elems contains a tuple which shows the variables used and a list of elements that match the constraint
        self._relations = elems[1]
        self._weight = weight

    def evaluate(self, variables, domains):
        currVal = []
        for var in self._vars:
            currVal.append(domains[var][variables[var]])
        for rel in self._relations:
            if rel == currVal:
                return self._weight
        return 0
    def getWeight(self):
        return self._weight
    
class Organism:
    def __init__(self, genome, constraints, prob, domains, fitOffset):    
        self._genome = genome
        self._domains = domains
        self._constraints = constraints
        self._fitOffset = fitOffset
        self._fitness = self._computeFitness()
        self._prob = prob
    
    def _computeFitness(self):
        solvedConstraints = 0
        constraintNum = len(self._constraints)
        for clause in self._constraints :
            solvedConstraints += clause.evaluate(self._genome, self._domains)
        return solvedConstraints + self._fitOffset         #fitness >= fitOffset (generally 1)
    
    def mutate(self):                        #triggers a mutation cycle where each gene can change with probability prob
        varNum = len(self._genome)
        for i in range(0, varNum) :
            doMutation = np.random.binomial(1, self._prob[i], 1)    #do the mutation with the probability of the respective gene
            if doMutation == 1:
                domain = self._domains[i]
                x = 1
                domLen = len(domain)
                if domLen > 2:                                      #in this case, we have to decide to which value of the domain we mutate
                    x = np.random.randint(1, domLen)                #sample an offset
                self._genome[i] = (x + self._genome[i]) % domLen
        self._fitness = self._computeFitness()
        
    def offspring(self, orgNum, label, dct):                    #returns the number of offspring and a label to identify their (common) genome
        myFitness = self._fitness               
        s = int(np.random.poisson(myFitness - 1, 1) + 1)   #draw a sample from Poisson distribution; we ensure that each organism has at least one offspring
        dct[label] = Organism(self._genome.copy(), self._constraints, self._prob, self._domains, self._fitOffset);
        return (s, label)       
    
    def getFitness(self):
        return self._fitness
    
    def getGenome(self):
        return self._genome.copy()
    
    def computeMutants(self) :   #returns a list of all the possible mutants of an organism at Hamming distance 1
        res = []
        l = len(self._genome)
        for i in range (0, l) :
            domain = self._domains[i]
            domLen = len(domain)
            for off in range (1, domLen):
                newGenome = self._genome.copy()
                newGenome[i] = (newGenome[i] + off) % domLen
                res.append(Organism(newGenome, self._constraints, self._prob, self._domains, self._fitOffset))
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
        self.plotAvgSel()
        
    def _fillPlot(self, extreme, middle):
        xs = [x[0] for x in extreme]  #fill between 2 lines
        ys1 = [x[1] for x in extreme]
        ys2 = [x[1] for x in middle]
        plt.fill_between(xs, ys1, ys2, color = "royalblue")
    
    def plotAvgSel(self):
        def expFunc(x, a, c, d):
            return a*np.exp(-c*x) + d
        def powFunc(x, m, c, c0):
            return c0 + x**m * c
        self._plotSelStat(self._avgSelList, expFunc, "Exponential","Rounds", 0)
        self._plotSelStat(self._avgSelList, powFunc, "Power law","Log rounds", 1)
        
    def getExpCoeff(self):
        def expFunc(x, a, c, d):
            return a*np.exp(-c*x) + d
        xs = [x[0] for x in self._avgSelList]
        ys = [x[1] for x in self._avgSelList]
        popt = self._compCoeff(xs,ys,expFunc)
        return popt[2]
    
    def getPowCoeff(self):
        def powFunc(x, m, c, c0):
            return c0 + x**m * c
        xs = [x[0] for x in self._avgSelList]
        ys = [x[1] for x in self._avgSelList]
        popt = self._compCoeff(xs,ys,powFunc)
        return popt[0]
    
    
    def _compCoeff(self, xs, ys, func):
        popt, pcov = curve_fit(func, xs, ys)
        return popt
    
    def _plotSelStat(self, statList, func, desc1 , desc2, k):     #plot the selection and fit the power law and exponential curves
        xs = [x[0] for x in statList]
        ys = [x[1] for x in statList]
        popt = self._compCoeff(xs,ys,func)
        plt.title("Selection coefficient statistics")
        plt.ylabel("Selection coefficient")
        plt.xlabel("Rounds")
        plt.plot(xs,ys, 'ro', label = "Average selection coefficient", color = "red")
        yy = [func(x, *popt) for x in xs]
        plt.plot(xs,yy, label = str(desc1) + " decay", color = "blue")
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        plt.title("Log selection coefficient statistics")
        plt.ylabel("Log selection coefficient")
        plt.xlabel(desc2)
        d = popt[2]                            #compute the offset
        mn = np.amin(ys)
        if d > mn:
            d = mn - 0.0000000001
        yy = [np.log(y - d) for y in ys]
        if k == 1:
            xs = xs[1:]
            yy = yy[1:]
            xr = [np.log(x) for x in xs]
            plt.plot(xr,yy, 'ro', label = "Average log selection coefficient minus offset", color = "red")
            m,b = np.polyfit(xr,yy,1)
            yr = [m * x + b for x in xr]
            plt.plot(xr, yr, label = "Log " + str(desc1)+ " decay", color = "blue")
        else:
            plt.plot(xs,yy, 'ro', label = "Average log selection coefficient minus offset", color = "red")
            m,b = np.polyfit(xs,yy,1)
            yr = [m * x + b for x in xs]
            plt.plot(xs, yr, label = "Log " + str(desc1)+ " decay", color = "blue")
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
        
        
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
    def __init__(self, orgNum, constraints, prob, initial, domains, fitOffset):
        self._orgNum = orgNum
        self._constraints = constraints
        self._organisms = []
        self._round = 0
        self._maxPossibleFitness = fitOffset    #used to normalize fitness when plotting
        
        for x in constraints:
            self._maxPossibleFitness += x.getWeight()

        self._stats = Statistics(self, self._maxPossibleFitness, self._orgNum)
        self._localStats = LocalStatistics(self, self._maxPossibleFitness)    

        for i in range(0, self._orgNum):
            self._organisms.append(Organism(initial, self._constraints, prob, domains, fitOffset))
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
        dct = {}                                                 #map a label to an organism; use it only after getting n from the next generation pool
        for i in range(0, self._orgNum):
            nextGenPool.append(self. _organisms[i].offspring(self._orgNum, i, dct))       #compute the next generation pool (of pairs (number of orgs, label) ) based on the fitness
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
                org = dct[label]
                org.mutate()                                     #mutate the offspring
                self._organisms.append(org)
        for i in range (0, totalRemaining):                      #add the remaining ones
        	org = dct[label]
        	org.mutate()									     #mutate them
        	self._organisms.append(org)          
        self._stats.updateStats()                                #update the stats for the new generation
        self._localStats.updateStats()


class Simulator:  
    def __init__(self,
    			probType,		 #the type of the problem solved (sat - 1, binary constraint - 2, general constraint - 3, wcsp definition - 4)
                initial,         #the initial values given to each variable / initial genome, expressed as a position in the domain
                probability,     #the probability of a mutation for each gene
                rounds,          #the number of rounds to be considered
                orgNum,          #the number of organisms
                constraints,     #the list of constraints
                domains,         #the list of domains, with a domain for each variable
                fitOffset):		 #the offset added to the fitness function
        self._rounds = rounds
        self._constraints = constraints
        self._genLength = len(initial)
        self._probType = probType
        if(probType == 1 or probType == 2):
        	domains = []
        	for i in range (0, len(initial)):
        		domains.append([0,1])     #for these problems (sat + binary constraint), define a binary domain
        self._population = Population(orgNum, constraints, probability, initial, domains, fitOffset)
    
    def run(self):
        for i in range (0, self._rounds):
            self._population.nextGeneration()    
    def printLocalStatistics(self):
    	self._population.getLocalStats().plotStats()
    	
    def printStatistics(self):
    	self._population.getStats().plotStats()
        
    def getExpCoeff(self):
        return self._population.getLocalStats().getExpCoeff()
    	
    def plotConstraintGraph(self):
        if self._probType != 2:
            print("Constraint graph available only for binary constraints")
            return nx.Graph()
        else:
            G = nx.Graph()
            G.add_node(0)
            draw(G, show = 'ipynb')
            for v in range (0, self._genLength):
                G.add_node(v)
            for cons in self._constraints:
                if cons.type() == "21":
                    elem = cons.getElem()
                    weight = cons.getWeights()
                    G.add_edge(elem, elem, label = str(weight), color = 'green')
                if cons.type() == "221":
                    elems = cons.getElems()
                    weights = cons.getWeights()
                    G.add_edge(elems[0], elems[1], label = str(weights), color = 'blue')
                if cons.type() == "221":
                    elems = cons.getElems()
                    weights = cons.getWeights()
                    G.add_edge(elems[0], elems[1], label = str(weights), color = 'red')
            return G
