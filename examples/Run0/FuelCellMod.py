from FuelCellSolver import FuelCellSolver
from PhysMod import PhysMod

class FuelCellMod(PhysMod):

  def __init__(self, *args, **kwargs):
    
    super().__init__(FuelCellSolver(), *args, **kwargs)

  def directSolve(self, case):

    case['beta'], case['obj'], case['ftrs'], lmbda = self.solver.solveForCaseWithBeta(
      case_id      = case['id'],
      beta         = case['beta'],
      nSolverIters = case['nSolverIters']
    )

    return lmbda
