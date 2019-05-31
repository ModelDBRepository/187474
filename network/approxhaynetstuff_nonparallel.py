# approxhaynetstuff_nonparallel
# A Python library for setting the (non-synaptic) parameters of the model network.
# Assumes a non-parallelized simulation, where the L5PCs are labeled as "cells.o[i]", i from 0 to Nmc-1
# Tuomo Maki-Marttunen, 2015-2016

from neuron import h
import matplotlib
matplotlib.use('Agg')
import numpy
from pylab import *

# params: a dictionary of model parameters
# Nmc: the number of cells
def setparams(params,Nmc):
  global dists_apical, dists_basal

  keys = params.keys()
  lengthChanged = False
  for ikey in range(0,len(keys)):
    key = keys[ikey]
    if key[0:2] == "L_":
      lengthChanged = True
    underscoreind = key.rfind('_')
    section = key[underscoreind+1:len(key)]
    if section == "*":
      for i in range(0,Nmc):
        h("""
i = """+str(i)+"""
forsec cells.o[i].all """+key[0:underscoreind]+""" = """+str(params[key])+"""
""")
      h("forall "+key[0:underscoreind]+" = "+str(params[key]))
    else:
      for i in range(0,Nmc):
        h("""
i = """+str(i)+"""
cells.o[i]."""+section+""" """+key[0:underscoreind]+""" = """+str(params[key])+"""
""")

  if lengthChanged:
    for i in range(0,Nmc):
      h("""
i = """+str(i)+"""
cells.o[i].soma diam = """+str(360.132/params['L_soma'])+"""
cells.o[i].dend diam = """+str(2821.168/params['L_dend'])+"""
cells.o[i].apic[0] diam = """+str(4244.628/params['L_apic[0]'])+"""
cells.o[i].apic[1] diam = """+str(2442.848/params['L_apic[1]'])+"""

lengthA = 0
lengthB = 0
forsec cells.o[i].apical {
  lengthA = lengthA + L
}
forsec cells.o[i].basal {
  lengthB = lengthB + L
}
cells.o[i].pA = lengthA/(lengthA + lengthB)
""")

