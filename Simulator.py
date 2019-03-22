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
    def __init__(self, genome, constraints, prob, domains, fitOffset, mutType):    
        self._genome = genome
        self._domains = domains
        self._constraints = constraints
        self._fitOffset = fitOffset
        self._fitness = self._computeFitness()
        self._prob = prob
        self._mutType = mutType
    
    def _computeFitness(self):
        solvedConstraints = 0
        constraintNum = len(self._constraints)
        for clause in self._constraints :
            solvedConstraints += clause.evaluate(self._genome, self._domains)
        return solvedConstraints + self._fitOffset         #fitness >= fitOffset (generally 1)
    
    def mutate(self):                        #triggers a mutation cycle where each gene can change with probability prob
        if self._mutType == 0:
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
        else:
            doMutation = np.random.binomial(1, self._prob, 1)      #decide if we mutate or not
            varNum = len(self._genome)
            if doMutation == 1:                                   #if yes, choose a gene to mutate
                i = np.random.randint(0, varNum)
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
        #s = myFitness
        dct[label] = Organism(self._genome.copy(), self._constraints, self._prob, self._domains, self._fitOffset, self._mutType)
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
                res.append(Organism(newGenome, self._constraints, self._prob, self._domains, self._fitOffset, self._mutType))
        return res


class LocalStatistics:     #computes local optimality statistics, regarding the fittest closest mutant
    def __init__(self, population, maxPossibleFitness, selOffset, genStat) :
        self._population = population
        self._maxFitList = []      #delta fitness of the fittest mutant after each round
        self._avgMutList = []      #average delta fitness of the possible mutants per round
        self._avgFitterList = []   #average delta fitness of the possible mutants that have higher fitness per round
        self._maxPossibleFitness = maxPossibleFitness   #used to normalize fitness before plotting
        self._normList = []     #used to display the normalization equivalent
        self._neighbourData = {}
        #self._avgSelListOld = []
        #self._avgSelListFitAvg = []
        self._avgSelListFitterAvg = []
        self._errorSelListFitterAvg = []
        self._avgSelListFittestAvg = []
        self._errorSelListFittestAvg = []
        self._selOffset = selOffset
        self._genStat = genStat
        #self._avgSelListOld50Per = []
        
    def updateStats(self) :
        rnd = self._population.getRound()
        roundNeighbourData = {}
        (maxDFit, avgMut, avgFitter) = self._computeStats(roundNeighbourData)
        self._neighbourData[rnd] = roundNeighbourData
        self._maxFitList.append((rnd, maxDFit))
        self._avgMutList.append((rnd, avgMut))
        self._avgFitterList.append((rnd, avgFitter))
        self._normList.append((rnd, 1 / self._maxPossibleFitness))
        #avgSelOld = self._computeSelectionOld(roundNeighbourData)  #compute the selection coefficients statistics
        #self._avgSelListOld.append((rnd, avgSelOld))
        #if rnd >= self._selOffset:
        #    avgSelFitAvg = self._computeSelectionFitAvg()
        #    self._avgSelListFitAvg.append((rnd, avgSelFitAvg))
        fitterRes = self._computeSelectionFitterAvg(roundNeighbourData)
        avgSelFitterAvg = fitterRes[0]
        errorSelFitter = fitterRes[1]
        self._avgSelListFitterAvg.append((rnd, avgSelFitterAvg))
        self._errorSelListFitterAvg.append((rnd, errorSelFitter))
        fittestRes = self._computeSelectionFittestAvg(roundNeighbourData)
        avgSelFittestAvg = fittestRes[0]
        errorSelFittest = fitterRes[1]
        self._avgSelListFittestAvg.append((rnd, avgSelFittestAvg))
        self._errorSelListFittestAvg.append((rnd, errorSelFittest))
        #avgSelOld50Per = self._computeSelectionOld50Per(roundNeighbourData)
        #self._avgSelListOld50Per.append((rnd, avgSelOld50Per))
     
        
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
        def expFunc(x, a, c):
            return a*np.exp(-c*x)
        def powFunc(x, m, c, c0):
            return c0 + x**m * c
        #self._plotSelStat(self._avgSelListOld, expFunc, "Exponential","Rounds", 0, "Local selection coefficient statistics")
        #self._plotSelStat(self._avgSelListOld, powFunc, "Power law","Log rounds", 1, "Local selection coefficient statistics")
        #self._plotSelStat(self._avgSelListFitAvg, expFunc, "Exponential","Rounds", 0, "Average fitness selection coefficient statistics")
        #self._plotSelStat(self._avgSelListFitAvg, powFunc, "Power law","Log rounds", 1, "Average fitness selection coefficient statistics")
        self._plotSelStat(self._avgSelListFitterAvg, self._errorSelListFitterAvg, expFunc, "Exponential","Rounds", 0, "Average fitter selection coefficient statistics")
        #self._plotSelStat(self._avgSelListFitterAvg, powFunc, "Power law","Log rounds", 1, "Average fitter selection coefficient statistics")
        self._plotSelStat(self._avgSelListFittestAvg, self._errorSelListFittestAvg, expFunc, "Exponential","Rounds", 0, "Average fittest selection coefficient statistics")
        #self._plotSelStat(self._avgSelListFittestAvg, powFunc, "Power law","Log rounds", 1, "Average fittest selection coefficient statistics")
        #self._plotSelStat(self._avgSelListOld50Per, expFunc, "Exponential","Rounds", 0, "Local 50% selection coefficient statistics")
        #self._plotSelStat(self._avgSelListOld50Per, powFunc, "Power law","Log rounds", 1, "Local 50% selection coefficient statistics")
        
    def getExpCoeff(self):
        def expFunc(x, a, c):
            return a*np.exp(-c*x)
        xs = [x[0] for x in self._avgSelListFitterAvg]
        ys = [x[1] for x in self._avgSelListFitterAvg]
        popt = self._compCoeff(xs,ys,expFunc)
        error = 0
        for (x,y) in zip(xs,ys):
            error += (y - expFunc(x, *popt))**2
        error = error / (2 * len(ys))
        return (popt[1],error)
    
    def getPowCoeff(self):
        def powFunc(x, m, c, c0):
            return c0 + x**m * c
        xs = [x[0] for x in self._avgSelListFitterAvg]
        ys = [x[1] for x in self._avgSelListFitterAvg]
        popt = self._compCoeff(xs,ys,powFunc)
        return popt[0]
    
    
    def _compCoeff(self, xs, ys, func, sigma = None):
        popt, pcov = curve_fit(func, xs, ys, sigma = sigma, absolute_sigma=True)
        return popt
    
    def _plotSelStat(self, statList, errorList, func, desc1 , desc2, k, title):     #plot the selection and fit the power law and exponential curves
        xs = [x[0] for x in statList]
        ys = [x[1] for x in statList]
        yerr = [x[1] for x in errorList]
        popt = self._compCoeff(xs,ys,func,[max(x,0.001) for x in yerr])
        plt.figure(figsize=(15,10))
        plt.title(title)
        plt.ylabel("Selection coefficient")
        plt.xlabel("Rounds")
        plt.errorbar(xs,ys, label = "Average selection coefficient", color = "red",markersize=5, yerr = yerr, barsabove = True, fmt = 'ro')
        yy = [func(x, *popt) for x in xs]
        plt.plot(xs,yy, label = str(desc1) + " decay", color = "blue")
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        plt.title("Error")
        plt.ylabel("Error values")
        plt.xlabel("Rounds")
        plt.plot(xs,yerr, 'ro', label = "error")
        plt.show()
        plt.title(title)
        plt.ylabel("Log selection coefficient")
        plt.xlabel(desc2)
        #d = popt[2]
        #print(d)
        
        if k == 1:
            res = [(np.log(x), np.log(y)) for (x,y) in filter(lambda t: t[0] > 0 and t[1] > 0, zip(xs, ys))]
            xr = [x[0] for x in res]
            yr = [x[1] for x in res]
            plt.plot(xr,yr, 'ro', label = "Average log selection coefficient", color = "red")
            #m,b = np.polyfit(xr,yy,1)
            #yr = [m * x + b for x in xr]
            #plt.plot(xr, yr, label = "Log " + str(desc1)+ " decay", color = "blue")
        else:
            res = [(x, np.log(y)) for (x,y) in filter(lambda t: t[1] > 0, zip(xs, ys))]
            xr = [x[0] for x in res]
            yr = [x[1] for x in res]
            plt.plot(xr,yr, 'ro', label = "Average log selection coefficient", color = "red")
            #m,b = np.polyfit(xs,yy,1)
            #yr = [m * x + b for x in xs]
            #plt.plot(xs, yr, label = "Log " + str(desc1)+ " decay", color = "blue")
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
            maxFit = 0
            sAvgFitterRaw = 0   #non-delta, computed for each organism
            nAvgFitterRaw = 0
            for x in mutants:
                xFit = x.getFitness()
                dFit = xFit - org.getFitness()
                mutantsDeltaFitness.append(dFit)
                if xFit > maxFit:
                    maxFit = xFit
                if dFit > maxDFit:
                    maxDFit = dFit
                sAvgMut += dFit
                nAvgMut += 1
                if dFit > 0:    #fitter than its parent
                    nAvgFitterRaw += 1
                    sAvgFitterRaw += xFit
                    sAvgFitter += dFit
                    nAvgFitter += 1
            fit = org.getFitness()
            if maxFit == 0:
            	maxFit = fit
            if nAvgFitterRaw == 0:
            	nAvgFitterRaw = 1
            	sAvgFitterRaw = fit
            roundNeighbourData[org] = (fit, mutantsDeltaFitness, maxFit, sAvgFitterRaw / nAvgFitterRaw)  #compute the neighbour data for the current round
        if nAvgFitter == 0:
            nAvgFitter = 1
        return (maxDFit, sAvgMut / nAvgMut, sAvgFitter / nAvgFitter)
                                
    #def _computeSelectionOld(self, roundNeighbourData):
    #    selection = []
    #    for org in self._population.getOrganisms():
    #        (fit, mutDeltaFit) = roundNeighbourData[org]
    #        maxMutDeltaFit = max(np.amax(mutDeltaFit),0)
    #        selection.append(maxMutDeltaFit / fit)     #compute the selection coefficient
    #    selection.sort()
    #    avgSel = np.mean(selection)
    #    return avgSel
        
    #def _computeSelectionOld50Per(self, roundNeighbourData):
    #    selection = []
    #    orgs = self._population.getOrganisms()
    #    orgs = sorted(orgs, key = lambda org : -org.getFitness())
    #    n = len(orgs)
    #    orgs = orgs[:int(n - n / 2)]
    #    for org in self._population.getOrganisms():
    #        (fit, mutDeltaFit) = roundNeighbourData[org]
    #        maxMutDeltaFit = max(np.amax(mutDeltaFit),0)
    #        selection.append(maxMutDeltaFit / fit)     #compute the selection coefficient
    #    selection.sort()
    #    avgSel = np.mean(selection)
    #    return avgSel
        
    #def _computeSelectionFitAvg(self):
    #    k = self._selOffset                      #number of steps that we look ahead
    #    rnd = self._population.getRound()
    #    avgFit = self._genStat.getAvgFitRaw()
    #    sel = (max(avgFit[rnd] - avgFit[rnd - k], 0)) / avgFit[rnd - k]
    #    return sel

    def _computeError(self, a, b, sa, sb, sab):
        return np.abs(a / b) * np.sqrt((sa / a) * (sa / a) + (sb / b) * (sb / b) - 2 * (sab) / (a * b))
    
    def _computeSelectionFitterAvg(self, roundNeighbourData):
        rnd = self._population.getRound()
        orgFit = []
        orgAvgFitter = []
        n = 0
        for org in self._population.getOrganisms():
            (fit, mutDFit, maxFit, avgFitter) = roundNeighbourData[org] 
            orgFit.append(fit)
            orgAvgFitter.append(avgFitter)
            n += 1
        sdFit = np.std(orgFit, ddof = 1)            #standard deviation of orgFit
        sdFitter = np.std(orgAvgFitter, ddof = 1)   #standard deviation of orgAvgFitter
        sdAvgFit = sdFit / (np.sqrt(n))         #standard deviation of the mean of orgFit
        sdAvgFitter = sdFitter / (np.sqrt(n))   #standard deviation of the mean of of orgAvgFitter
        avgFit = np.mean(orgFit)                    #mean of orgFit
        avgFitter = np.mean(orgAvgFitter)           #mean of orgAVgFitter
        covFitFitter = np.cov([orgFit,orgAvgFitter], ddof = 1)           #covariance between orgFit and orgAvgFitter
        covAvgFitFitter = 1 / (n) * covFitFitter                    #covariance between the means                 
        sel = (max(avgFitter - avgFit,0)) / avgFit
        err = self._computeError(avgFitter, avgFit, sdAvgFitter, sdAvgFit, covAvgFitFitter[0][1])
        return (sel, err)
        
    def _computeSelectionFittestAvg(self, roundNeighbourData):
        rnd = self._population.getRound()
        orgFit = []
        orgFittest = []
        n = 0
        for org in self._population.getOrganisms():
            (fit, mutDFit, maxFit, avgFitter) = roundNeighbourData[org] 
            orgFit.append(fit)
            orgFittest.append(maxFit)
            n += 1
        sdFit = np.std(orgFit, ddof = 1)             #standard deviation of orgFit
        sdFittest = np.std(orgFittest, ddof = 1)   #standard deviation of orgFittest
        sdAvgFit = sdFit / (np.sqrt(n))          #standard deviation of the mean of orgFit
        sdAvgFittest = sdFittest / (np.sqrt(n))  #standard deviation of the mean of of orgFittest
        avgFittest = np.mean(orgFittest)                #mean of orgFittest
        avgFit = np.mean(orgFit)                     #mean of orgFit
        covFitFittest = np.cov(orgFit, orgFittest, ddof = 1)           #covariance between orgFit and orgFittest
        covAvgFitFittest = 1 / (n) * covFitFittest                 #covariance between the means   
        sel = (max(avgFittest - avgFit,0)) / avgFit
        err = self._computeError(avgFittest, avgFit, sdAvgFittest, sdAvgFit, covAvgFitFittest[0][1])
        return (sel, err)
    
                                
    def getNeighbourData(self):
        return self._neighbourData;    #this contains the local data (neighbour data) for a round, and organism, its fitness and its possible mutants fitness
    
    #For the visualisation of this:
    #Show data for a specific round / show data using joyplots (might not be easy to understand)
    #How to visualize the data for the population: 3D graph / joyplots / show separate distribution for organisms and then for mutants for each organism


class Statistics:                       #contains all the statistics and methods to update and plot them for a population
    def __init__(self, population, maxPossibleFitness, orgNum) :
        self._population = population
        self._avgFitList = []
        self._avgFitRaw = []   #unnormalized fitness
        self._maxFitList = []
        self._minFitList = []
        self._max5PerFitList = []
        self._min5PerFitList = []
        self._maxPossibleFitness = maxPossibleFitness   #used to normalize fitness before plotting
        self._orgNum = orgNum
        
    def updateStats(self):
        rnd = self._population.getRound()
        avgFit = self._computeAvgFit()
        self._avgFitRaw.append(avgFit)
        self._avgFitList.append((rnd, avgFit / self._maxPossibleFitness))
        self._maxFitList.append((rnd, self._computeMaxFit() / self._maxPossibleFitness))
        self._minFitList.append((rnd, self._computeMinFit() / self._maxPossibleFitness))
        self._max5PerFitList.append((rnd, self._computeMaxPerFit(5) / self._maxPossibleFitness))
        self._min5PerFitList.append((rnd, self._computeMinPerFit(5) / self._maxPossibleFitness))
        
        
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
    
    def _computeMinPerFit(self, per):
        per = per / 100
        orgs = self._population.getOrganisms()
        fit = []
        for i in range(0, self._orgNum):
            fit.append(orgs[i].getFitness())
        fit.sort()
        l = int(math.ceil(len(fit) * per))      #get only the first 5%
        fit = fit[:l]
        s = 0
        for x in fit:                            #compute the average of it
            s += x
        return s / l
    
    def _computeMaxPerFit(self, per):
        per = per / 100
        orgs = self._population.getOrganisms()
        fit = []
        for i in range(0, self._orgNum):
            fit.append(orgs[i].getFitness())
        fit.sort(reverse = True)
        l = int(math.ceil(len(fit) * per))      #get only the first 5% (biggest 5%)
        fit = fit[:l]
        s = 0
        for x in fit:                            #compute the average of it
            s += x
        return s / l
    
    def plotGenome(self):
        orgs = self._population.getOrganisms()
        for org in orgs:
            print(str(org.getFitness()) + "   " + str(org.getGenome()))
            
    def getAvgFitRaw(self):
    	return self._avgFitRaw


class Population:                        #contains the current population and statistics; is updated on nextGeneration call
    def __init__(self, orgNum, constraints, prob, initial, domains, fitOffset, mutType):
        self._orgNum = orgNum
        self._constraints = constraints
        self._organisms = []
        self._round = 0
        self._maxPossibleFitness = fitOffset    #used to normalize fitness when plotting
        
        for x in constraints:
            self._maxPossibleFitness += x.getWeight()

        self._stats = Statistics(self, self._maxPossibleFitness, self._orgNum)
        self._localStats = LocalStatistics(self, self._maxPossibleFitness, 3, self._stats)    

        for i in range(0, self._orgNum):
            self._organisms.append(Organism(initial, self._constraints, prob, domains, fitOffset, mutType))
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
        (num, label) = nextGenPool[-1]
        for i in range (0, totalRemaining):                      #add the remaining ones
        	org = dct[label]
        	org.mutate()									     #mutate them
        	self._organisms.append(org)          
        self._stats.updateStats()                                #update the stats for the new generation
        self._localStats.updateStats()

    def computeDistribution(self):
        distrib = {}
        for org in self._organisms:
            fit = org.getFitness()
            distrib[fit] = distrib.get(fit, 0) + 1
        return distrib


class Simulator:  
    def __init__(self,
    			probType,		 #the type of the problem solved (sat - 1, binary constraint - 2, general constraint - 3, wcsp definition - 4)
                initial,         #the initial values given to each variable / initial genome, expressed as a position in the domain
                probability,     #the probability of a mutation for each gene
                rounds,          #the number of rounds to be considered
                orgNum,          #the number of organisms
                constraints,     #the list of constraints
                domains,         #the list of domains, with a domain for each variable
                fitOffset,		 #the offset added to the fitness function
                mutType,         #the type of mutations implemented
                getDistrib = False):     #true, if we want the distributions to be computed
        self._rounds = rounds
        self._constraints = constraints
        self._genLength = len(initial)
        self._probType = probType
        self._distrib = []
        self._getDistrib = getDistrib
        if(probType == 1 or probType == 2):
        	domains = []
        	for i in range (0, len(initial)):
        		domains.append([0,1])     #for these problems (sat + binary constraint), define a binary domain
        self._population = Population(orgNum, constraints, probability, initial, domains, fitOffset, mutType)
    def run(self):
        for i in range (0, self._rounds):
            if self._getDistrib:
                self._distrib.append(self._population.computeDistribution())
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
                if cons.type() == "222":
                    elems = cons.getElems()
                    weights = cons.getWeights()
                    G.add_edge(elems[0], elems[1], label = str(weights), color = 'red')
            return G
    
    def writeRunDataToFile(self, f):
        stats = self._population.getStats()
        localStats = self._population.getLocalStats()
        f.write("Fitness:\n")
        f.write(str(stats._avgFitList))
        f.write("\n")
        f.write("Selection coefficient:\n")
        f.write(str(localStats._avgSelListFitterAvg))
        f.write("\n")
        f.write("\n")
    
    def getDistribution(self):
        if not self._getDistrib:
            print("Distribution by fitness not computed!")
        return self._distrib
