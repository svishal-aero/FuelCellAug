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
    ]
})

#A = np.loadtxt('metric_log.txt')

A_temp = []
i = 0
while os.path.exists('performance_logs/metrics.%d' % i):
  A_temp.append(np.loadtxt('performance_logs/metrics.%d' % i))
  i += 1
A = np.array(np.vstack(A_temp))

print("%d/%d cases showed better or equal performance" % (A[A[:,1]>=0.5].shape[0], A.shape[0]))

m00 = 0.0
m10 = 1.0 / (2 - 0.10) - 0.5
m25 = 1.0 / (2 - 0.25) - 0.5
m50 = 1.0 / (2 - 0.50) - 0.5
m90 = 1.0 / (2 - 0.90) - 0.5

alpha=1.0
plt.hist(A[:,1]-0.5, bins=42, orientation='horizontal', color='grey')
plt.plot([A[0,0]-1, A[-1,0]+1], [m00, m00], '-k', linewidth=3.0, alpha=alpha, label='Equal RMS error')
plt.plot([A[0,0]-1, A[-1,0]+1], [m10, m10], '-r', linewidth=3.0, alpha=alpha, label='-10% error')
plt.plot([A[0,0]-1, A[-1,0]+1], [m25, m25], '-g', linewidth=3.0, alpha=alpha, label='-25% error')
plt.plot([A[0,0]-1, A[-1,0]+1], [m50, m50], '-b', linewidth=3.0, alpha=alpha, label='-50% error')
plt.xlabel('Number of cases')
plt.ylabel('Performance metric')
plt.xlim(left=A[0,0]-1, right=80)
plt.ylim(bottom=-0.5, top=0.5)
plt.legend()
plt.tight_layout(pad=1.01)
plt.savefig('histogram.png')
plt.show()

alpha=0.3
plt.bar(A[A[:,1]>=0.5,0], 2*A[A[:,1]>=0.5,1]-1, width=1, color='green')
plt.bar(A[A[:,1]<0.5,0], 2*A[A[:,1]<0.5,1]-1, width=1, color='red')
#plt.plot([A[0,0]-1, A[-1,0]+1], [m00, m00], '-k', linewidth=3.0, alpha=alpha, label='Equal RMS error')
#plt.plot([A[0,0]-1, A[-1,0]+1], [m10, m10], '-r', linewidth=3.0, alpha=alpha, label='-10% error')
#plt.plot([A[0,0]-1, A[-1,0]+1], [m25, m25], '-g', linewidth=3.0, alpha=alpha, label='-25% error')
#plt.plot([A[0,0]-1, A[-1,0]+1], [m50, m50], '-b', linewidth=3.0, alpha=alpha, label='-50% error')
plt.xlabel('Case IDs')
plt.ylabel(r'Performance metric ($\mathscr{P}_1$) for water content predictions')
plt.xlim(left=A[0,0]-1, right=A[-1,0]+1)
plt.ylim(bottom=-1, top=1)
plt.legend()
plt.tight_layout(pad=1.01)
plt.savefig('performance_water1_SparseTraining.png')
plt.show()
