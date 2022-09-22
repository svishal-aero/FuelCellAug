import numpy as np

A = np.loadtxt("metric_log.txt")[:,1]

ind1 = np.argsort(A)
ind2 = np.argsort(np.abs(A-0.5))

print(ind1[-6:])
print(ind1[:6])
print(ind2[:6])
