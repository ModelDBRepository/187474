#drawfits_withmids_combfs
#A script that illustrates the differences between reconstructed and reduced morphology
#Tuomo Maki-Marttunen, 2015-2016
#
#Running the script:
#  python drawmorphs.py filename
#
#Arguments:
#  filename
#    The input filename with the parameter names and values, e.g., "pars_withmids_combfs_0a_1a_2a_3a" (loads "pars_withmids_combfs_0a_1a_2a_3a.sav"), 
#    by default "pars_withmids_combfs_final"
#
#Input files needed:
#
#  snmf_protocols.py:
#    Fitting protocols (which quantities are fitted at each step; what stimuli are for each objective function; etc.)
#  mytools.py:
#    General tools for e.g. spike detection
#
#Output files:
#
#  morph_<<filename>>.eps: Figure illustrating the reconstructed and reduced morphology
#

import matplotlib
matplotlib.use('Agg')
from pylab import *
import pickle
from neuron import h

filename = 'pars_withmids_combfs_final'

if len(sys.argv) > 1: 
  filename = sys.argv[1]


def boxoff(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


paramdict = {}

unpicklefile = open(filename+".sav", 'r')
unpickledlist = pickle.load(unpicklefile)
unpicklefile.close()
par_names = unpickledlist[0]
par_values = unpickledlist[1]
for i in range(0,len(par_names)):
  paramdict[par_names[i]] = par_values[i]

close("all")
f,axarr = subplots(1,1)

h("""
load_file("stdlib.hoc")
load_file("stdrun.hoc")
objref cvode           
cvode = new CVode()    
cvode.active(1)        
cvode.atol(0.00005)    
load_file("import3d.hoc")
objref L5PC_orig
load_file(\"models/L5PCbiophys3.hoc\")
load_file(\"models/L5PCtemplate.hoc\")
L5PC_orig = new L5PCtemplate(\"morphologies/cell1.asc\")
access L5PC_orig.soma
""")
#Tuft, apic, soma, basal
cols = ["#FF0000", "#008899", "#FFAA00", "#0000FF"]

for itree in range(0,3):
  if itree == 0:
    nsec = len(h.L5PC_orig.dend)
  elif itree == 1:
    nsec = len(h.L5PC_orig.apic)
  else:
    nsec = 1
  for j in range(nsec-1,-1,-1):
    if itree == 0:
      h("access L5PC_orig.dend["+str(j)+"]")
    elif itree == 1:
      h("access L5PC_orig.apic["+str(j)+"]")
    else:
      h("access L5PC_orig.soma")
    h("tmpvarx = x3d(0)")
    h("tmpvary = y3d(0)")
    h("tmpvarz = z3d(0)")
    h("tmpvarx2 = x3d(n3d()-1)")
    h("tmpvary2 = y3d(n3d()-1)")
    h("tmpvarz2 = z3d(n3d()-1)")
    coord1 = [h.tmpvarx,h.tmpvary,h.tmpvarz]
    coord2 = [h.tmpvarx2,h.tmpvary2,h.tmpvarz2]
    col = cols[0]
    if itree == 0:
      col = cols[3]
    elif 0.5*(coord1[1]+coord2[1]) < 650:
      col = cols[1]
    if itree == 2:
      col = cols[2]
    h("""
myn = n3d()
myx0 = x3d(0)
myy0 = y3d(0)
myz0 = z3d(0)
""")
    oldcoord = [h.myx0, h.myy0, h.myz0]
    for k in range(1,int(h.myn)):
      h("""
myx0 = x3d("""+str(k)+""")
myy0 = y3d("""+str(k)+""")
myz0 = z3d("""+str(k)+""")
mydiam = diam""")
      axarr.plot([oldcoord[0],h.myx0],[oldcoord[1],h.myy0],'k-',linewidth=h.mydiam*0.25,color=col)
      oldcoord = [h.myx0, h.myy0, h.myz0]
axis("equal")

somacoord = oldcoord

len_names = ['L_apic[1]', 'L_apic[0]', 'L_soma', 'L_dend']
lens = []
for i in range(0,len(len_names)):
  for j in range(0,len(par_names)):
    if par_names[j]==len_names[i]:
      lens.append(par_values[j])
      break
diams = [2442.848/lens[0],4244.628/lens[1],360.132/lens[2],2821.168/lens[3]]

xoffset = 400
ystarts = [somacoord[1]+lens[1]+lens[2],somacoord[1]+lens[2],somacoord[1],somacoord[1]]
ydirs = [1,1,1,-1]
for i in range(0,len(lens)):
  axarr.plot([xoffset,xoffset],[ystarts[i],ystarts[i]+ydirs[i]*lens[i]],linewidth=diams[i]*0.25,color=cols[i])

boxoff(axarr)
axarr.set_xticks([-200,0,200,400])
axarr.set_xlabel("x (um)")
axarr.set_ylabel("y (um)")
axarr.set_position([0.125,0.1,0.2,0.8])
#axis([-362,562,-350,2300])
axis([-250,450,-550,1450])
axarr.text(-520,1350,'A',fontsize=40)
f.savefig("morph_"+filename+".eps")

