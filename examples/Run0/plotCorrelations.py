import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    'figure.figsize':[12,5],
    'font.size':15,
    'text.usetex':True,
    'text.latex.preamble':[
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
        r'\usepackage{mathrsfs}',
    ],
})

ids = np.linspace(1, 1224, 1224)

def plotPerf(perf):
    print(perf[perf >= 0].size)
    plt.bar(ids[perf >= 0], perf[perf >= 0], width=1, color='green', alpha=0.3)
    plt.bar(ids[perf <  0], perf[perf <  0], width=1, color='red', alpha=0.3)

perfList = []
i = 0
while os.path.exists('performance_logs/metrics.%d' % i):
    perfList.append(np.loadtxt('performance_logs/metrics.%d' % i))
    i += 1
A = np.array(np.vstack(perfList))

metric_1_water = (A[:,1] - A[:,2]) / (A[:,1] + A[:,2])
metric_2_water = metric_1_water * A[:,5] / (A[:,1] + A[:,2])
metric_1_current = (A[:,3] - A[:,4]) / (A[:,3] + A[:,4])
metric_2_current = metric_1_current * A[:,6] / (A[:,3] + A[:,4])

B = np.loadtxt('inletValues.txt')

def getScaledInput(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

plotPerf(metric_1_water)
plt.plot(ids, getScaledInput(B[:,0]), label='Scaled Cell Current Density', color='black')
plt.plot(ids, getScaledInput(B[:,2]), label='Scaled Cathode Stoichiometry', color='red')
plt.plot(ids, getScaledInput(B[:,3]), label='Scaled Anode Relative Humidity', color='blue')
plt.plot(ids, getScaledInput(B[:,4]), label='Scaled Channel Inlet Temperature', color='green')
for i in [40, 100, 125, 155, 190, 230, 400, 685, 740, 840, 865, 1000, 1090, 1200]:
    plt.plot([i,i], [-1, 2], '--k')
plt.xlabel('Case IDs')
plt.ylabel('Qualitative trends')
plt.xlim(left=0, right=1225)
plt.ylim(bottom=-0.5, top=1.05)
plt.title(r'Qualitative analysis of the augmentation function')
plt.legend(ncol=2)
plt.tight_layout(pad=0.5)
plt.savefig('Analysis1')
plt.show()

plotPerf(metric_1_water)
plt.plot(ids, getScaledInput(B[:,9]-B[:,7]), label='Scaled anode inlet-outlet pressure difference', color='black')
plt.xlabel('Case IDs')
plt.ylabel('Qualitative trends')
plt.xlim(left=0, right=1225)
plt.ylim(bottom=-0.5, top=1.05)
plt.title(r'Qualitative analysis of the augmentation function')
plt.legend(ncol=2)
plt.tight_layout(pad=0.5)
plt.savefig('Analysis2')
plt.show()
