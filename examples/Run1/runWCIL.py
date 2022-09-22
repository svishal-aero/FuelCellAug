import sys
sys.path.append('/home/vsriv/pyMAPS_prev/')
import numpy as np

from mpi4py import MPI
from FuelCellMod import FuelCellMod
from AugFn.KerasNN import KerasNN
from WeaklyCoupledInferenceAndLearning import WCIL

model = FuelCellMod()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

augFn = KerasNN(
  'lambdaEq',
  structure = [len(model.solver.feature_name_list), 7, 7, 1],
  actFn = ['linear', 'sigmoid', 'sigmoid', 'relu'],
  learning_rate = 1e-3,
  batchSize = 24,
  nTrainIters = 500,
)

initMode = 'f'
if rank==0:
  initMode = input("How do you wish to initialize the neural network? (f: Fresh init, p: previous init, r: restart init): ")

initMode = comm.bcast(initMode, root=0)

if initMode=='f': startFrom = 0

elif initMode=='p':
  startFrom = 0
  for i in range(size):
    if i==rank:
      augFn.load(identifier='init')
  comm.Barrier()

elif initMode=='r':
  startFrom = np.loadtxt('obj.dat', skiprows=1).shape[0]
  for i in range(size):
    if i==rank:
      augFn.load(identifier='%04d' % (startFrom-1))
  comm.Barrier()

#case_id_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
#case_id_list = [40, 100, 125, 155, 190, 230, 400, 685, 740, 840, 865, 1000, 1090, 1200]
case_id_list = [40, 125, 190, 400, 740, 865, 1090]
cases = [{'id':case_id, 'obj':0.0, 'ftrs':None, 'beta':None, 'sens':None, 'nSolverIters':100} for case_id in case_id_list]

if comm.rank==0: print(augFn.params)

WCIL(
  model,
  augFn,
  nIters = 200,
  cases = cases,
  alpha = 0.02,
  startFrom = startFrom
)
