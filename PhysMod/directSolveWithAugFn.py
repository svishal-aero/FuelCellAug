import sys
import numpy as np

def directSolveWithAugFn(physMod, augFn, case, nAugIters, absTol, relax):

  relax_default = relax

  # Solve for the baseline solution
  physMod.directSolve(case)
  beta_temp = case['beta']
  physMod.writeLog("AugIter %03d    AugResid %le    Objective %le"%(0, 0.0, case['obj']))

  simulationHasNotConverged = True

  i = 0

  while True:

    try:

      # Obtain new augmentation field
      beta_predict = augFn.predict(case['ftrs'], runOnProcs='ALL')
      case['beta'] = relax * beta_predict + (1.0-relax) * beta_temp

      # Obtain augmentation residual
      augResid = np.linalg.norm(beta_predict - beta_temp)

      # Check for relaxation factor limit
      if relax<1e-6:
        physMod.writeErr("WARNING: Relaxation factor is less than 1e-6 (virtually zero)!! Stopping iterations...")
        break

      # Solve with updated augmentation field and set the output augmentation field to 'beta' instead of 'betaNew'
      physMod.directSolve(case)
      beta_temp = case['beta']

      # Print augmentation residual
      physMod.writeLog("AugIter %03d    AugResid %le    Objective %le"%(i+1, augResid, case['obj']))
      i += 1

      # Check for convergence
      if augResid < absTol:
        simulationHasNotConverged = False
        break

      # Check for maximum number of iterations
      if i==nAugIters: break

      # Set relaxation factor to preset
      relax = relax_default

    except KeyboardInterrupt:
      sys.exit(0)

    except:
      
      # Reduce relaxation factor to half and print message to terminal
      relax /= 2
      physMod.writeLog("Reduced relaxation factor to %le"%(relax))

      # Continue loop
      continue

  # Display warning if not converged
  if simulationHasNotConverged:
    physMod.writeErr("WARNING: Augmentation residual did not converge to the set tolerance of %le"%(absTol))
