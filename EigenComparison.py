# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:42:16 2023

@author: bradc
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

#################################################################################
#################################################################################

# Read the current directory for eigenvalue files with the specified precision produced by the pspec2 program
def readfiles(p):
    files = [f for f in os.listdir() if "jEigenvals" in f and "P" + str(p) + ".txt" in f]
    data_out = {}
    for file in files:
        fdata = []
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Ignore commented lines
                if '#' in line:
                    pass
                elif len(line) == 1:
                    pass
                else:
                    # Remove white space and outer brackets
                    #line = line.strip()[1:-1]
                    # Read real and imaginary parts into numpy data
                    fdata.append(complex(float(line[0]),float(line[1])))
        # Polynomial order = len / 2
        N = len(fdata) / 2
        data_out[N] = sorted(fdata, key = lambda x : (x.real)**2 + (x.imag)**2)
    return data_out

# Search for the closest eigenvalues between the specified resolutions Nlow
# and Nhigh, and return those eigenvalues. If the closest
# value from a different resolution is further than the next nearest
# eigenvalue of the lowest resolution, then return a complex distance
def closest(data, Nlow, Nhigh):

    # Default value for Nlow
    if Nlow not in data.keys():
        print("ERROR: couldn't find low resolution N =", Nlow, "data from",
              sorted(data.keys()))
        Nlow = sorted(data.keys())[0]
        print("Defaulting to lowest resolution data with N =", Nlow)

    # Default value for Nhigh
    if Nhigh not in data.keys():
        print("ERROR: couldn't find high resolution N =", Nhigh, "data from",
              sorted(data.keys()))
        Nhigh = sorted(data.keys())[-1]
        print("Defaulting to highest resolution data with N =", Nhigh)

    # Get the respective data
    lrdata = data.get(Nlow)
    hrdata = data.get(Nhigh)
    close = []

    # Iterate through low res data and compare to high res data. Store
    # the matched eigenvalues and the distances between them if
    # applicable
    for i in range(len(lrdata)):
        # Set the maximum distance betweeen eigenvalues
        maxsep = 0.
        if i == 0:
            maxsep = abs(lrdata[i] - lrdata[i+1])
        elif i + 1 >= len(lrdata):
            maxsep = abs(lrdata[i] - lrdata[i-1])
        else:
            maxsep = min(abs(lrdata[i]-lrdata[i-1]),
                         abs(lrdata[i]-lrdata[i+1]))
        # Find the nearest eigenvalue in the highres data
        distance = 1.E5
        indx = len(hrdata)
        for j in range(len(hrdata)):
            this_dis = abs(lrdata[i] - hrdata[j])
            if this_dis < distance:
                distance = this_dis
                indx = j
            else:
                pass

        # If minimum distance exceeds nearest neighbour distance,
        # return complex value. If no near neighbour was found,
        # return complex value
        if distance >= maxsep or indx == len(hrdata) - 1:
            close.append([lrdata[i], None, 0. + 1.j])
        else:
            close.append([lrdata[i], hrdata[indx], distance])

    # Return the array of eigenvalues, nearest neighbours, and distances
    return close

#################################################################################
#################################################################################

def main(p_in, *args):
    # Find available data
    data = readfiles(p_in)
    # Compare min/max resolutions to inputs
    Nlow = sorted(data.keys())[0]
    Nhigh = sorted(data.keys())[-1]
    if len(args) == 0:
        pass
    elif len(args) == 1:
        if float(args[0][0]) > Nlow:
            Nlow = int(args[0][0])
        else:
            pass
    else:
        if min(args) > Nlow:
            Nlow = min(args)
        if max(args) < Nhigh:
            Nhigh = max(args)

    print("Reading", p_in, "bit eigenvalues with resolutions:", sorted(data.keys()))
    for key in sorted(data.keys()):
        if key >= Nlow and key <= Nhigh:
            print("N =", key)
            #print(data.get(key))
        else:
            pass


    print("Comparing eigenvalues from N =", Nlow, "to N =", Nhigh)
    nearest = closest(data, Nlow, Nhigh)
    print("Closest matches:")
    for x in nearest:
        print(x)

    # Compare the convergence of the first Nmin/2
    # eigenvalues between lowest and specified resolution
    # See Boyd (7.19)-(7.20) for procedure
    odif = []
    for i in range(int(Nlow/2)):
        # Calculate weights based on values of adjacent eigenvalues
        if i == 0:
            # If values are pure imaginary/pure real, they can't be compared
            # to general complex values
            if (nearest[0][0].real == 0.0 and nearest[0][1].real == 0.0) or \
                (nearest[0][0].imag == 0.0 and nearest[0][1].imag == 0.0):
                wt = min(abs(nearest[0][0]), abs(nearest[0][1]))
            else:
                l1, l2 = nearest[0][:2]
                wt = abs(nearest[0][0]-nearest[0][1])
        else:
            # If values are pure imaginary/pure real, they can't be compared
            # to general complex values
            if (nearest[i][0].real == 0.0 and nearest[i][1].real == 0.0) or \
                (nearest[i][0].imag == 0.0 and nearest[i][1].imag == 0.0):
                wt = min(abs(nearest[i][0]), abs(nearest[i][1]))
            else:
                l1, l2, l3 = nearest[i-1:i+2][0]
                wt = 0.5 * (abs(l2-l1) + abs(l3-l2))

        # Ignore if the distance between low and high res eigenvalues is complex
        #if np.iscomplex(nearest[i][2]):
        #    pass
        #else:
        odif.append(abs(nearest[i][0] - nearest[i][1]) / wt)

        if (1./odif[i]) > 1E3 or odif[i] == 0.:
                print("Good eigenvalue:", nearest[i], 1./odif[i])

    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(10,6))
    this_label = r'$N =$ ' + str(int(Nlow)) + r' vs ' + str(int(Nhigh)) + r' at ' + str(p_in) + r' bits of precision'
    plt.plot([i for i in range(1,int(len(odif)+1))], [1./val for val in odif],
            'x', label=this_label)
    plt.ylabel(r'$\sigma^{-1}_{nearest}$')
    plt.xlabel(r'$N$')
    plt.yscale('log')
    plt.legend()
    plt.savefig('data/NearestEigDiff_N' + str(Nlow) + 'N' + str(Nhigh) + 'P' + str(p_in) + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    plt.show()

#################################################################################
#################################################################################

if len(sys.argv) < 2:
    print("To run: python EigenComparison.py P Nmin Nmax")
    print("\t P (int): bits of precision that appear in eigenvalue file names. All eigenvalue files with Nmin to Nmax with precision P will be compared")
    print("\t Nmin (int): minimum spectral resolution value for comparison")
    print("\t Nmax (int): maximum spectral resolution value for comparison")
    print("")
elif len(sys.argv) == 2:
    main(sys.argv[1])
else:
    main(sys.argv[1], sys.argv[2:])

#################################################################################
#################################################################################





























