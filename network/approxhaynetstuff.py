# approxhaynetstuff_nonparallel
# A Python library for setting the (non-synaptic) parameters of the model network.
# Assumes a parallelized simulation, where the L5PCs are labeled as "epnm.pc.gid2cell(i)", i from 0 to Nmc-1
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
if (epnm.gid_exists(i)) {
  forsec epnm.pc.gid2cell(i).all """+key[0:underscoreind]+""" = """+str(params[key])+"""
}
""")
      h("forall "+key[0:underscoreind]+" = "+str(params[key]))
    else:
      for i in range(0,Nmc):
        h("""
i = """+str(i)+"""
if (epnm.gid_exists(i)) {
  epnm.pc.gid2cell(i)."""+section+""" """+key[0:underscoreind]+""" = """+str(params[key])+"""
}
""")

  if lengthChanged:
    for i in range(0,Nmc):
      h("""
i = """+str(i)+"""
if (epnm.gid_exists(i)) {
    epnm.pc.gid2cell(i).soma diam = """+str(360.132/params['L_soma'])+"""
    epnm.pc.gid2cell(i).dend diam = """+str(2821.168/params['L_dend'])+"""
    epnm.pc.gid2cell(i).apic[0] diam = """+str(4244.628/params['L_apic[0]'])+"""
    epnm.pc.gid2cell(i).apic[1] diam = """+str(2442.848/params['L_apic[1]'])+"""

    lengthA = 0
    lengthB = 0
    forsec epnm.pc.gid2cell(i).apical {
      lengthA = lengthA + L
    }
    forsec epnm.pc.gid2cell(i).basal {
      lengthB = lengthB + L
    }
    epnm.pc.gid2cell(i).pA = lengthA/(lengthA + lengthB)
}
""")
