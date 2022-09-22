import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size':20, 'figure.figsize':[10, 8]})

A = np.loadtxt('log_1000.txt')
I = np.linspace(1, A.size, A.size)
plt.semilogy(I, A, '-ob')
plt.xlabel('Augmentation iterate')
plt.ylabel('Augmentation residual')
plt.tight_layout(pad=1.01)
plt.savefig('figures/iterAug')
plt.show()
