import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size':16, 'figure.figsize':[8,6]})
import matplotlib.pyplot as plt

obj = np.loadtxt("obj.dat",skiprows=1)
plt.plot(obj[:,0],obj[:,1],'-ob')
plt.xlabel('Itrerations')
plt.ylabel('Combined objective')
plt.tight_layout(pad=1.01)
plt.savefig('figures/obj.png')
plt.show()
