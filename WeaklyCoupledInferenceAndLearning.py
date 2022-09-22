import sys
from mpi4py import MPI
import numpy as np

def WCIL(model, augFn, nIters=10, cases=[], alpha=0.1, startFrom=0):
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  if rank==0:
    if startFrom==0:
      open('obj.dat','w').write('Iteration    Combined Objective\n')
    else:
      model.writeLog('Restarting from iteration %d'%(startFrom))
      for case in cases:
        case['beta'] = np.loadtxt("./beta_restart/%s.dat"%(str(case['id'])))[:,np.newaxis]

  ftrs_all = []
  beta_all = []
  obj_all  = 0.0

  for case in cases:
    if rank==0:
      model.writeLog("\nSolving case id %s"%(str(case['id'])))
      model.directSolve(case)
      ftrs_all.append(case['ftrs'])
      beta_all.append(case['beta'])

  for i in range(startFrom, nIters):

    if rank==0:
      ftrs_all = np.vstack(ftrs_all)
      beta_all = np.vstack(beta_all)
      print('', flush=True)

    augFn.train(ftrs_all, beta_all, verbose=1)
    augFn.save(identifier='%04d' % i)
    comm.Barrier()

    ftrs_all = []
    beta_all = []
    obj_all = 0.0
    
    for case in cases:
        
      if rank==0: model.directSolveWithAugFn(augFn, case)
      
      # Obtain sensitivities w.r.t. this converged beta field
      comm.Barrier()
      model.solveAndObtainBetaSens(case)
      comm.Barrier()
      
      # Update beta field
      # NOTE: There is a limiter implemented here that limits beta to non-negative values only
      if rank==0 and case['sens'] is not None:
        case['beta'] -= alpha * case['sens'] / np.max(np.abs(case['sens']))
        case['beta'][case['beta']<0.0] = 0.0
      
      # Solve to obtain corresponding features with the updated beta field
      if rank==0:
        model.writeLog("\nSolving case id %s"%(str(case['id'])))
        model.directSolve(case)
        ftrs_all.append(case['ftrs'])
        beta_all.append(case['beta'])
      
      # Save restart file
      if rank==0: np.savetxt('./beta_restart/%s.dat'%(str(case['id'])), case['beta'])

      # Objective function with the updated beta field
      obj_all += case['obj']

    # Write objective value to prompt and to file
    if rank==0: model.writeLog('Iteration = %3d    Combined Objective = %le'%(i, obj_all))
    if rank==0: open('obj.dat','a').write('%9d    %-le\n'%(i, obj_all)) 
