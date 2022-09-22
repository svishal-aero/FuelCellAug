import os
import sys
import numpy as np

from .directSolveWithAugFn   import directSolveWithAugFn
from .solveAndObtainBetaSens import solveAndObtainBetaSens

class PhysMod:

  def __init__(self, solver, adjoint_solver=None, logFile=sys.stdout, errFile=sys.stderr):

    self.solver         = solver
    self.adjoint_solver = adjoint_solver

    self.logFile        = logFile
    self.errFile        = errFile

    self.writeLog('\n----------------------------------------------------------------------------------------------------------------------')
    self.writeLog('\n  Physical model initialized')
    self.writeLog('\n----------------------------------------------------------------------------------------------------------------------')

  


  def directSolveWithAugFn(self, augFn, case, nAugIters=100, absTol=1e-3, relax=0.5, useRestart=True):
    if os.path.exists('./beta_restart/%s.dat'%(str(case['id']))) and useRestart:
      self.writeLog('Loading pre-calculated beta from restart file "./beta_restart/%s.dat"'%(str(case['id'])))
      case['beta'] = np.loadtxt('./beta_restart/%s.dat'%(str(case['id'])))[:,np.newaxis]
    directSolveWithAugFn(self, augFn, case, nAugIters, absTol, relax)
    self.writeLog('Writing restart file "./beta_restart/%s.dat"'%(str(case['id'])))
    np.savetxt('./beta_restart/%s.dat'%(str(case['id'])), case['beta'])

  


  def directSolve(self, case):
    pass

  


  def directAndAdjointSolve(self, case):
    pass

  


  def solveAndObtainBetaSens(self, case, fd_step=1e-4):
    solveAndObtainBetaSens(self, case, fd_step)




  def writeLog(self, message):
    self.logFile.write('%s\n'%(message))
    self.logFile.flush()




  def writeErr(self, message):
    if self.errFile==sys.stderr:
      self.errFile.write('\u001b[31m%s\u001b[0m\n'%(message))
    else:
      self.errFile.write('%s\n'%(message))
    self.errFile.flush()
