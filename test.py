from Simulator import *

initial = [0, 0, 0, 0, 0, 0, 0, 0]
probability = 0.5
rounds = 50
orgNum = 100
clause1 = Clause([1],1)
clause2 = Clause([2],1)
clause3 = Clause([3],1)
clause4 = Clause([4],1)
clause5 = Clause([5],1)
clause6 = Clause([6],1)
clause7 = Clause([7],1)
clause8 = Clause([8],1)
clauses = [clause1, clause2, clause3, clause4, clause5, clause6, clause7, clause8]

mySim = Simulator(initial, probability, rounds, orgNum, clauses)
mySim.run()
