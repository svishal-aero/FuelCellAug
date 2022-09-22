import sys
sys.path.append('/home/vsriv/pyMAPS_prev/')

from subprocess import call
import numpy as np

from FuelCellMod import FuelCellMod
from AugFn.KerasNN import KerasNN

import matplotlib as mpl
mpl.rcParams.update({'figure.figsize':[10,6]})
#mpl.rcParams.update({'font.size':30})
import matplotlib.pyplot as plt
plt.rc('font',       size=25) #controls default text size
plt.rc('axes',  titlesize=25) #fontsize of the title
plt.rc('axes',  labelsize=25) #fontsize of the x and y labels
plt.rc('xtick', labelsize=25) #fontsize of the x tick labels
plt.rc('ytick', labelsize=25) #fontsize of the y tick labels
plt.rc('legend', fontsize=25) #fontsize of the legend

for id in [40, 100, 125, 155, 230, 400, 685, 740, 840, 865, 1000, 1090, 1200, 250, 450, 500, 730, 920, 1215, 110, 600, 811, 847, 1023, 1066, 203, 452, 693, 52, 1030, 1199]:

    model = FuelCellMod()
    nFtrs = len(model.solver.feature_name_list)
    
    augFn = KerasNN(
      'lambdaEq',
      structure=[nFtrs,7,7,1],
      actFn=['linear', 'sigmoid', 'sigmoid', 'relu']
    )
    
    augFn.load(identifier='0023')
    
    case={
      'id':id,
      'obj':0.0,
      'ftrs':None,
      'beta':None,
      'sens':None,
      'nSolverIters':100
    }
    
    data = model.solver.data.get_outlets(case['id'])['Lambda Distribution [-]'].flatten()
    
    baseline = model.directSolve(case)
    currDist = np.loadtxt("currDist_%04d.txt" % case['id'])
    data_current = currDist[:,0]
    baseline_current = currDist[:,1]
    
    model.directSolveWithAugFn(
      augFn,
      case,
      nAugIters=3,
      relax=0.3,
      absTol=1e-3,
      useRestart=True
    )
    
    augmented = model.directSolve(case)
    currDist = np.loadtxt("currDist_%04d.txt" % case['id'])
    augmented_current = currDist[:,1]
    
    call("rm currDist_%04d.txt" % case['id'], shell=True)
    
    model.writeLog("Objective Function = %le" % (case['obj']))
    
    y = np.linspace(0,1,20)
    
    plt.figure()
    plt.plot(y, data, 'ok', label='Data')
    plt.plot(y, baseline,'-r', linewidth=3, label='Baseline')
    plt.plot(y, augmented,'-g', linewidth=3, label='Augmented')
    plt.title('Case %d'%(id))
    plt.xlabel('y')
    plt.ylabel('Lambda distribution')
    plt.legend()
    plt.tight_layout(pad=0.3)
    plt.savefig('figures/%s.png'%(str(case['id'])))
    #plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(y, data_current, 'ok', label='Data')
    plt.plot(y, baseline_current,'-r', linewidth=3, label='Baseline')
    plt.plot(y, augmented_current,'-g', linewidth=3, label='Augmented')
    plt.title('Case %d'%(id))
    plt.xlabel('y')
    plt.ylabel('Current distribution')
    plt.legend()
    plt.tight_layout(pad=0.3)
    plt.savefig('figures/%s_c.png'%(str(case['id'])))
    #plt.show()
    plt.close()
