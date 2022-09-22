import sys
sys.path.append('/home/vsriv/FuelCellAugmentation/')

import numpy as np

from FuelCellMod import FuelCellMod
from AugFn.KerasNN import KerasNN

import matplotlib as mpl
mpl.rcParams.update({'figure.figsize':[10,6]})
#mpl.rcParams.update({'font.size':30})
import matplotlib.pyplot as plt
plt.rc('font',       size=15) #controls default text size
plt.rc('axes',  titlesize=15) #fontsize of the title
plt.rc('axes',  labelsize=15) #fontsize of the x and y labels
plt.rc('xtick', labelsize=15) #fontsize of the x tick labels
plt.rc('ytick', labelsize=15) #fontsize of the y tick labels
plt.rc('legend', fontsize=15) #fontsize of the legend

model = FuelCellMod()
nFtrs = len(model.solver.feature_name_list)

augFn = KerasNN(
  'lambdaEq',
  structure=[nFtrs,7,7,1],
  actFn=['linear', 'sigmoid', 'sigmoid', 'relu']
)

augFn.load(identifier='%04d' % int(sys.argv[2]))

case={
  'id':int(sys.argv[1]),
  'obj':0.0,
  'ftrs':None,
  'beta':None,
  'sens':None,
  'nSolverIters':100
}

data = model.solver.data.get_outlets(case['id'])['Lambda Distribution [-]'].flatten()

baseline = model.directSolve(case)
currDist = np.loadtxt("currDist.txt")
data_current = currDist[:,0]
baseline_current = currDist[:,1]

# PLOT RESIDUALS ====================================================================

for k, v in model.solver.sim.built_model.rhs.items():
    print(k, v.size)

residual = {}
for k in model.solver.sim.built_model.rhs.keys():
    residual[k] = []

for i in range(len(model.solver.sim.solution.t)):
    for k, v in model.solver.sim.built_model.rhs.items():
        residual[k].append(
            np.linalg.norm(
                v.evaluate(model.solver.sim.solution.t[i],
                           model.solver.sim.solution.y.full()[:,i],
                           inputs=model.solver.sim.solution.all_inputs[-1]
                          )
            )
        )

plt.figure(figsize=[15,6])        

for k in model.solver.sim.built_model.rhs.keys():
    if str(k)[:6] == 'concat': labelStr = 'Water Content (Anode CL, Cathode CL, Membrane)'
    else: labelStr=k
    plt.semilogy(model.solver.sim.solution.t, residual[k], '-o', label=labelStr)

plt.legend(bbox_to_anchor=[1.02,0.5], loc='center left')
plt.xlabel('Physical time')
plt.ylabel('Residual value')
plt.tight_layout(pad=1.01)
plt.savefig('figures/iterSolve.png')
plt.show()

#====================================================================================

model.directSolveWithAugFn(
  augFn,
  case,
  nAugIters=100,
  relax=0.3,
  absTol=1e-3,
  useRestart=True
)

augmented = model.directSolve(case)
currDist = np.loadtxt("currDist.txt")
augmented_current = currDist[:,1]

model.writeLog("Objective Function = %le" % (case['obj']))

y = np.linspace(0,1,20)

plt.plot(y, data, 'ok', label='Data')
plt.plot(y, baseline,'-r', linewidth=3, label='Baseline')
plt.plot(y, augmented,'-g', linewidth=3, label='Augmented')
plt.title('Case %d'%(int(sys.argv[1])))
plt.xlabel('y')
plt.ylabel('Lambda distribution')
plt.legend()
plt.tight_layout(pad=1.01)
plt.savefig('figures/%s.png'%(str(case['id'])))
plt.show()

model.directSolve(case)

plt.plot(y, data_current, 'ok', label='Data')
plt.plot(y, baseline_current,'-r', linewidth=3, label='Baseline')
plt.plot(y, augmented_current,'-g', linewidth=3, label='Augmented')
plt.title('Case %d'%(int(sys.argv[1])))
plt.xlabel('y')
plt.ylabel('Current distribution')
plt.legend()
plt.tight_layout(pad=1.01)
plt.savefig('figures/%s_c.png'%(str(case['id'])))
plt.show()
