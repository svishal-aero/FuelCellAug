import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    'figure.figsize':[12,6],
    'font.size':15,
    'text.usetex':True,
    'text.latex.preamble':[
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
        r'\usepackage{mathrsfs}',
    ],
})

def plotPerf(name, ids, perf, title=''):
    print(perf[perf >= 0].size)
    plt.bar(ids[perf >= 0], perf[perf >= 0], width=1, color='green')
    plt.bar(ids[perf <  0], perf[perf <  0], width=1, color='red'  )
    plt.xlabel('Case IDs')
    plt.ylabel('Performance metric')
    plt.xlim(left=0, right=1225)
    plt.ylim(bottom=-0.5, top=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig('performance_%s.png' % name)
    plt.show()

perfList = []
i = 0
while os.path.exists('performance_logs/metrics.%d' % i):
    perfList.append(np.loadtxt('performance_logs/metrics.%d' % i))
    i += 1
A = np.array(np.vstack(perfList))

ids = A[:,0]

metric_1_water = (A[:,1] - A[:,2]) / (A[:,1] + A[:,2])
metric_2_water = metric_1_water * A[:,5] / (A[:,1] + A[:,2])
plotPerf('water1', ids, metric_1_water, r'Performance metric 1 ($\mathscr{P}_1$) for water content predictions')
plotPerf('water2', ids, metric_2_water, r'Performance metric 2 ($\mathscr{P}_2$) for water content predictions')

metric_1_current = (A[:,3] - A[:,4]) / (A[:,3] + A[:,4])
metric_2_current = metric_1_current * A[:,6] / (A[:,3] + A[:,4])
plotPerf('current1', ids, metric_1_current, r'Performance metric ($\mathscr{P}_1$) for current density predictions')
plotPerf('current2', ids, metric_2_current, r'Performance metric ($\mathscr{P}_2$) for current density predictions')

#B = np.loadtxt('inletValues.txt')
#A = A[np.argsort(B[:,3])]
#ids = np.linspace(1,1224,1224)
