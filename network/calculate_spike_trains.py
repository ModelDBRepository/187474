# Python file for running the network (N=150) simulations and saving the resulting spike trains
# Tuomo Maki-Marttunen, 2015-2016
import mytools
import simseedburst_func_nonparallel_nonadaptive_allions
from pylab import *
from neuron import h
import pickle
import sys

Nmc = 150
gSynCoeffs = [1.1, 1.25, 1.4]
gNoiseCoeff = 1.11
counter = -1
for igsyn in range(0,3):
  for myseed in range(1,15):
    counter = counter + 1
    if len(sys.argv) > 1 and int(sys.argv[1]) != counter:
      continue
    gSynCoeff = gSynCoeffs[igsyn]
    Q = simseedburst_func_nonparallel_nonadaptive_allions.simseedburst_func(Nmc, 11000, myseed, 0.0004, 0.001, 5, 1.0, gNoiseCoeff, gSynCoeff)
    picklelist = Q[2][:]
    file = open('spikes_nonadaptive_'+str(Nmc)+'_gsyn'+str(gSynCoeff)+'_seed'+str(myseed)+'.sav', 'w')
    pickle.dump(picklelist,file)
    file.close()
