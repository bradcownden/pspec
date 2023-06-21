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

###############################################################################
###############################################################################

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


###############################################################################
###############################################################################

# Specify spectral order
Nspec = 60

files = [f for f in os.listdir("./") if "j" in f and ".txt" in f]
efile = [f for f in files if "jEigenvals" in f][-1]
sfile = [f for f in files if "jpspec" in f][-1]

# Get eigenvalues and pseudospectra at increasing precision
data1 = readEigs('jEigenvals_N' + str(Nspec) + 'P64.txt')
data2 = readEigs('jEigenvals_N' + str(Nspec) + 'P128.txt')
data3 = readEigs('jEigenvals_N' + str(Nspec) + 'P256.txt')
sigma1, pdata1 = readSigma('jpspec_N' + str(Nspec) + 'P64.txt')
sigma2, pdata2 = readSigma('jpspec_N' + str(Nspec) + 'P128.txt')
sigma3, pdata3 = readSigma('jpspec_N' + str(Nspec) + 'P256.txt')

# Make sure the psuedospectra are evaluated at the same points on the grid
if np.any([(pdata1[i] - pdata2[i]) != 0. for i in range(len(pdata1))]) \
    or np.any([(pdata3[i] - pdata2[i]) != 0. for i in range(len(pdata2))]):
    print("\nERROR: Pseudospectrum characteristics do not match")
    for p in [pdata1, pdata2, pdata3]:
        print("xmin: %f, xmax: %f, ymin: %f, ymax: %f, grid: %f" %\
              (p[0], p[1], p[2], p[3], p[4]))
    print("")
else:
    D1 = np.linalg.norm(sigma3 - sigma2)
    D2 = np.linalg.norm(sigma2 - sigma1)
    if D2 == 0. or D1 == 0.:
        print("Pspec order of convergence: inf")
    else:
        print("Order of convergence of pspec:", np.log2(D1/D2))
    print("Pseudospectrum min/max:", np.min(sigma3), '/', np.max(sigma3))
    print("")


# Compare the convergence of the first Nmin/2
# eigenvalues between highest and lowest fitting
# See Boyd (7.19)-(7.20)
Nmin = len(data1)/4
Nmax = len(data3)/4
odif, good_vals, eigs = [], [], []
for i in range(int(Nmax)-1):
    # Calculate weighting based on the difference of nearby values
    if i == 0:
        wt = abs(data1[i] - data1[i+1])
    else:
        l1, l2, l3 = data1[i-1:i+2]
        wt = 0.5 * (abs(l2-l1) + abs(l3-l2))

    odif.append(abs(data3[i] - data1[i]) / wt)
    # Report 'good' eigenvalues
    if odif[i] == 0.0:
        print("Totally convergent eigenvalue:", data1[i])
    else:
        if (1./odif[i]) > 1E8:
            good_vals.append([i, 1./odif[i]])
            eigs.append([i, data1[i]])
            print("Good eigenvalue: i =", i, "w =", data1[i])

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(2,1, figsize=(8,8))
ax[0].plot(np.arange(len(odif)), [1./val for val in odif], 'C4x')
vshift = 1
def switch(i):
    return -1 if i % 2 == 0 else 1

for pairs in good_vals:
    ax[0].annotate(r'$\omega_{!s}$'.format('{' + str(pairs[0]) + '}'),
                xy=pairs, xycoords ='data', horizontalalignment='center',
                xytext=(0, switch(pairs[0]) * 30), textcoords='offset points',
                arrowprops=dict(facecolor='C4', edgecolor='None',
                width=1, headlength=4, headwidth=4, shrink=0.1),
                bbox=dict(pad=2, facecolor='None', edgecolor='None'))
    vshift += 1
ax[0].set_xlabel(r'$i$')
ax[0].set_ylabel(r'$\sigma^{-1}_{diff}$')
ax[0].set_yscale('log')
ax[0].set_ylim(10E-2, 10E17)

# Include a table of good eigenvalues
collabel=(r'Re $\omega_i$', r'Im $\omega_i$')
ax[1].axis('off')
ax[1].table(cellText=[[x[1].real, x[1].imag] for x in eigs],
            colLabels=collabel, loc='center')
plt.subplots_adjust(hspace=0.15)
plt.suptitle(r'$N = $ ' + str(Nspec))
plt.savefig('data/OrdinalEigDiff_N%d' % Nspec + '.pdf', format='pdf',
            transparent=True, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(10,8))
gs = GridSpec(2,2)
ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[0,1]))
ax.append(fig.add_subplot(gs[1,:]))


for this_ax in ax[:2]:
    this_ax.plot([e.real for e in data1], [e.imag for e in data1], 'C1.',
               markersize=30, alpha=0.33)
    this_ax.plot([e.real for e in data2], [e.imag for e in data2], 'C2.',
               markersize=20, alpha=0.66)
    this_ax.plot([e.real for e in data3], [e.imag for e in data3], 'C3.',
                 markersize=6)
    this_ax.grid(ls='--', alpha=0.5)
    this_ax.set_xlabel(r'Re $\omega$')
    this_ax.set_ylabel(r'Im $\omega$')

# Zoom in on sub-region
ax[1].set_xlim(-.55,.55)
ax[1].set_ylim(-.55,.55)

# Pseudospectrum
[X,Y] = np.mgrid[pdata3[0]:pdata3[1]:-1j*(pdata3[-1]+1),
                 pdata3[2]:pdata3[3]:-1j*(pdata3[-1]+1)]


# Plot the pseudospectrum: a filled contour plot; a set of contour lines
# at the desired powers of ten; a colourbar with log values in
# scientific notation
#levels = [1e-8 * 10 ** (i/3) for i in range(0,19)]
levels = [1e-6 * 10 ** (i/3) for i in range(0,19)]
CS = ax[2].contourf(X, Y, sigma3, levels=levels, locator=ticker.LogLocator(),
                    cmap=cm.viridis_r)
ax[2].contour(X, Y, sigma3, levels=levels, colors='white', linewidths=0.5)
cb = fig.colorbar(CS, ax=ax[2],
                    format=ticker.LogFormatterSciNotation(base=10.0))
cb.set_label(label=r'      $\sigma^\epsilon$', rotation='horizontal')
#ax[1].xaxis.set_major_formatter(ticker.LogLocator())
#ax[1].yaxis.set_major_formatter(ticker.LogLocator())
ax[2].plot([e.real for e in data3], [e.imag for e in data3], 'r.',
           markersize=6)
ax[2].set_xlabel(r'Re $\omega$')
ax[2].set_ylabel(r'Im $\omega$')
ax[2].set_xlim(pdata3[0],pdata3[1])
ax[2].set_ylim(pdata3[2],pdata3[3])
plt.suptitle(r'$N =$ ' + str(Nspec))
plt.savefig('data/pspec_N%d' % Nspec + '.pdf', format='pdf',
            transparent=True, bbox_inches='tight')
plt.show()
