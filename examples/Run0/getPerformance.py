import os, sys
import numpy as np

perfList = []
i = 0
while os.path.exists('performance_logs/metrics.%d' % i):
    perfList.append(np.loadtxt('performance_logs/metrics.%d' % i))
    i += 1
A = np.array(np.vstack(perfList))

id = int(sys.argv[1]) - 1

metric_1_water = (A[id,1] - A[id,2]) / (A[id,1] + A[id,2])
metric_2_water = metric_1_water * A[id,5] / (A[id,1] + A[id,2])
metric_1_current = (A[id,3] - A[id,4]) / (A[id,3] + A[id,4])
metric_2_current = metric_1_current * A[id,6] / (A[id,3] + A[id,4])

print(metric_1_water, metric_2_water)
print(metric_1_current, metric_2_current)
