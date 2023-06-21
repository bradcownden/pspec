# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:02:22 2023

@author: bradc
"""
import shutil as sh
import subprocess as sp
import numpy as np
import os

def updateInputs(Nnew):
    # Make a backup just in case
    try:
        sh.copyfile('Inputs.txt', 'Inputs.bk')
        #print("Quiet copy")
    except:
        print("\nERROR: Couldn't copy Inputs.txt to backup file. Aborting...\n")
        return 0
    # Read the file and update
    with open("Inputs.txt", 'r') as f:
        args = []
        for line in f.readlines():
            args.append(line.strip())
        # Change the spectral number (always the first line read)
        if 'spectral_N' in args[0]:
            args[0] = args[0].split('=')[0] + '=' + str(Nnew)

    # Write the data back into the file
    try:
        with open("Inputs.txt", 'w') as f:
            for arg in args:
                f.write(arg + '\n')
    except:
        # In the case of an error, restore the backup
        sh.copyfile('Inputs.bk', 'Inputs.txt')
        print("\nERROR: Couldn't write updated Inputs.txt file. Aborting...\n")
        return 0

    # Remove the backup
    try:
        os.remove('Inputs.bk')
        #print("Quite delete")
    except:
        print("\nERROR: Couldn't remove temporary backup file.")





prec = [str(64 * 2 ** i) for i in range(1,3) ]
res = [10 * i for i in range(7,11)]

# Run the julia program pspec.jl with increasing precisions over
# the set of resolutions. After each resolution has run at precisions
# 1, 96, 1042, increment the resolution in the inputs file and repeat
# the process. Run with a number of processors for all resolutions
# and precisions

print("Starting pseudospectrum calculations for resolutions " +\
      "N = %d to N = %d for %d to %d bits of precision" % (res[0], res[-1],\
      prec[0], prec[-1]))

for N in res:
    updateInputs(N)
    print("Current resolution:", N)
    for p in prec:
        print("Running with", p, "bits of precision")
        sp.check_output("julia -t auto ./pspec.jl " + p, shell=True)





























