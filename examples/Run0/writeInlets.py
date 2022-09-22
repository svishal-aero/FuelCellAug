import numpy as np
from FuelCellSolver import FuelCellSolver as FCS

fcs = FCS()
inletNames = []
for name in fcs.data.get_inlets(1):
    inletNames.append(name)

with open('inletNames.txt', 'w') as f:
    for name in inletNames:
        f.write(name+'\n')

inletValues = np.zeros((1224,len(inletNames)))
for i in range(1,1225):
    for j in range(len(inletNames)):
        inletValues[i-1,j] = fcs.data.get_inlets(i)[inletNames[j]]

np.savetxt('inletValues.txt', inletValues)
