# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:11:41 2023

@author: bradc
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.gridspec import GridSpec

def readEigs(file):
    # First line is the length of the vector of eigenvalues
    with open(file, 'r') as f:
        data = []
        N = 0
        for line in f.readlines():
            l = line.strip().split('\t')
            if len(l) == 1:
                N = int(l[0])
            else:
                data.append([float(l[0]),float(l[1])])
        # Turn into array of complex values
        out = np.ndarray((N,),dtype=complex)
        for i in range(len(data)):
            out[i] = data[i][0] + data[i][1]*1j

        return out

def readSigma(file):
    with open(file, 'r') as f:
        sig = []
        xmin, xmax, ymin, ymax, grid = 0.,0.,0.,0.,0.
        # First line is the bounds of the pseudospectrum and number of
        # grid points
        for line in f.readlines():
            l = line.strip().split('\t')
            if len(l) == 5:
                xmin = float(l[0])
                xmax = float(l[1])
                ymin = float(l[2])
                ymax = float(l[3])
                grid = float(l[4])
            elif len(l) == 1:
                N = int(l[0][1:-1].split(',')[0])
            else:
                sig.append([float(val) for val in l])
        out = np.ndarray((N,N))
        for i in range(len(sig)):
            out[i,:] = sig[i]
        return out, [xmin, xmax, ymin, ymax, grid]

files = [f for f in os.listdir("./") if "j" in f and ".txt" in f]
efile = [f for f in files if "jEigenvals" in f][-1]
sfile = [f for f in files if "jpspec" in f][-1]

data1 = readEigs('jEigenvals_N20.txt')
data2 = readEigs('jEigenvals_N25.txt')
data3 = readEigs('jEigenvals_N30.txt')
sigma, pdata = readSigma('jpspec_N20.txt')


# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(figsize=(10,8))
gs = GridSpec(2,2)
ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[0,1]))
ax.append(fig.add_subplot(gs[1,:]))

ax[0].plot([e.real for e in data1], [e.imag for e in data1], 'b.',
           markersize=14, alpha=0.5)
ax[0].plot([e.real for e in data2], [e.imag for e in data2], 'g.',
           markersize=12, alpha=0.6)

for this_ax in ax[:2]:
    this_ax.plot([e.real for e in data3], [e.imag for e in data3], 'r.')
    this_ax.grid(ls='--', alpha=0.5)
    this_ax.set_xlabel(r'Re $\omega$')
    this_ax.set_ylabel(r'Im $\omega$')

# Zoom in on sub-region
ax[1].set_xlim(-1,1)
ax[1].set_ylim(-1,1)

# Pseudospectrum
[X,Y] = np.mgrid[pdata[0]:pdata[1]:-1j*(pdata[-1]+1),
                 pdata[2]:pdata[3]:-1j*(pdata[-1]+1)]


# Plot the pseudospectrum: a filled contour plot; a set of contour lines
# at the desired powers of ten; a colourbar with log values in
# scientific notation
levels = [1e-3 * 2 ** i for i in range(0,11)]
CS = ax[2].contourf(X, Y, sigma, levels=levels, locator=ticker.LogLocator(),
                    cmap=cm.viridis_r)
ax[2].contour(X, Y, sigma, levels=levels, colors='white', linewidths=0.5)
cb = fig.colorbar(CS, ax=ax[2],
                    format=ticker.LogFormatterSciNotation(base=10.0))
cb.set_label(label=r'  $\sigma^\epsilon$', rotation='horizontal')
#ax[1].xaxis.set_major_formatter(ticker.LogLocator())
#ax[1].yaxis.set_major_formatter(ticker.LogLocator())
ax[2].plot([e.real for e in data3], [e.imag for e in data3], 'r.',
           markersize=6)
ax[2].set_xlabel(r'Re $\omega$')
ax[2].set_ylabel(r'Im $\omega$')
ax[2].set_xlim(pdata[0],pdata[1])
ax[2].set_ylim(pdata[2],pdata[3])

plt.show()
