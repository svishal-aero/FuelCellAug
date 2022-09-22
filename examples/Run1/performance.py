import sys
import numpy as np
from subprocess import call
from mpi4py import MPI

sys.path.append('/home/vsriv/FuelCellAugmentation/')
from FuelCellMod import FuelCellMod
from AugFn import KerasNN

import matplotlib as mpl
mpl.rcParams.update({'figure.figsize':[10,6]})
#mpl.rcParams.update({'font.size':30})
import matplotlib.pyplot as plt
plt.rc('font',       size=20) #controls default text size
plt.rc('axes',  titlesize=20) #fontsize of the title
plt.rc('axes',  labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=20) #fontsize of the legend

##########################################################################################################

augId = 32

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
beg = rank * 1224 // size + 1
end = (rank+1) * 1224 // size + 1 if rank < size-1 else 1225

if rank==0: call('rm -f performance_logs/*', shell=True)
comm.Barrier()

fileHandle = open('performance_logs/log.%d' % rank, 'w')

model = FuelCellMod(logFile=fileHandle, errFile=fileHandle)
nFtrs = len(model.solver.feature_name_list)

augFn = KerasNN(
  'lambdaEq',
  structure=[nFtrs,7,7,1],
  actFn=['linear', 'sigmoid', 'sigmoid', 'relu']
)

for i in range(size):
  if i==rank:
    augFn.load(identifier='%04d' % augId)
  comm.Barrier()

comm.Barrier()

for i in range(beg, end):
  
  model.writeLog('\nRunning Case %04d' % i)
  model.writeLog('-------------------')
  case={
    'id':int(i),
    'obj':0.0,
    'ftrs':None,
    'beta':None,
    'sens':None,
    'nSolverIters':100
  }
  
  data = model.solver.data.get_outlets(case['id'])['Lambda Distribution [-]'].flatten()
  
  baseline = model.directSolve(case)
  
  model.directSolveWithAugFn(
    augFn,
    case,
    nAugIters=100,
    relax=0.3,
    absTol=1e-3
  )
  
  augmented = model.directSolve(case)

  error_baseline = np.linalg.norm(baseline-data)
  error_augmented = np.linalg.norm(augmented-data)
  metric = error_baseline / (error_baseline + error_augmented)

  with open('performance_logs/metrics.%d' % rank, 'a') as f:
    f.write('%4d %.10le\n' % (i, metric))
  
  model.writeLog("Objective Function = %le" % (case['obj']))

fileHandle.close()
