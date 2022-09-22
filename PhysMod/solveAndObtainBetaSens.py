from mpi4py import MPI
import numpy as np

def solveAndObtainBetaSens(physMod, case, fd_step):

  # If adjoint solver is available, use it!
  
  if physMod.adjoint_solver is not None:

    physMod.directAndAdjointSolve(case)

  # If no adjoint solver is present, use finite differences in parallel
  
  else:
  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Solve the baseline simulation to obtain objective and features

    physMod.writeLog('\nEvaluating case ID ' + str(case['id']))
    
    if rank==0:
      physMod.directSolve(case)
    else:
      case['obj']  = 0
      case['ftrs'] = None
      case['beta'] = None

    # Communicate augmentation field and objective to all processors
    
    obj_base     = comm.bcast(case['obj'], root=0)
    case['beta'] = comm.bcast(case['beta'], root=0)
    
    # Initialize sensitivities
    
    nBeta        = case['beta'].size
    case['sens'] = np.zeros_like((case['beta']))

    # Number of passes required for all sensitivities
    
    nPasses = nBeta // size

    # Number of evaluations in an extra pass (if required)
    
    finalPass = nBeta % size

    # Increment number of passes by one if an extra pass is required
    
    if finalPass>0: nPasses += 1

    # Iterate over all passes
    
    for iPass in range(nPasses):
      
      # Index of beta being processed on this processor
      
      iBeta = iPass * size + rank

      # Find the largest index of beta being processed in this pass
      
      endBetaID = iBeta
      
      if rank==0:
        if iPass==nPasses-1 and finalPass>0:
          endBetaID = iBeta + finalPass - 1
        else:
          endBetaID = iBeta + size - 1

      # Calculate perturbed objectives
      
      physMod.writeLog("Solving for sensitivities %d/%d to %d/%d"%(iBeta+1, nBeta, endBetaID+1, nBeta))

      if iBeta<nBeta:
        case['beta'][iBeta,0] += fd_step
        physMod.directSolve(case)
        case['beta'][iBeta,0] -= fd_step
      else:
        case['obj'] = 0.0

      # Communicate perturbed objectives
      
      objArr = np.array(comm.allgather(case['obj']))
      
      # Fill sensitivities for this pass on the root processor
      
      if rank==0:
        case['sens'][iBeta:endBetaID+1,0] = (objArr[:endBetaID-iBeta+1]-obj_base) / fd_step

    case['obj'] = obj_base
