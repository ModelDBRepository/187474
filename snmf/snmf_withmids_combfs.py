#Simpilified neuron model fitter
#_withmids: consider not only best parameter sets for each objective but also "midoptimal" parameter sets that
#  perform well in two objectives
#_combfs: Combine certain objective functions to reduce the number of considered parameter set tree branches
#Tuomo Maki-Marttunen, 2015-2016
#
#Emoo code based on scripts from https://projects.g-node.org/emoo/
#  (Bahl et al. 2012, Automated optimization of a reduced layer 5 pyramidal cell model based on experimental data)
#
#
#
#Running the script:
#  python snmf_withmids_combfs.py ITER myseed
#or in parallel with e.g. 100 CPUs:
#  mpirun -np 100 nrniv -python -mpi snmf_withmids_combfs.py ITER myseed
#
#Arguments:
#  ITER:
#    Fitting task number, see below
#  myseed:
#    RNG seed (used by Emoo)
#
#Files needed:
#
#  Python files:
#
#    snmf_protocols.py:
#      Fitting protocols (which quantities are fitted at each step; what stimuli are for each objective function; etc.)
#    mytools.py:
#      General tools for e.g. spike detection
#
#  Data files:
#    originalrun.sav
#      Target data (generated from the full model based on snmf_protocols.py)
#    If 0 < ITER <= 3, one of the following files is needed (the parameter values obtained from the first step fitting)
#      pars_withmids_combfs_0a.sav
#      pars_withmids_combfs_1a.sav
#      pars_withmids_combfs_2a.sav
#    If 3 < ITER <= 12, one of the following files is needed (the parameter values obtained from the first and second step fitting)
#      pars_withmids_combfs_0a_0a.sav
#        ...
#      pars_withmids_combfs_2a_2a.sav
#    If 12 < ITER <= 66, one of the following files is needed (the parameter values obtained from the first, second and third step fitting)
#      pars_withmids_combfs_0a_0a_0a.sav
#       ...
#      pars_withmids_combfs_2a_2a_5a.sav
#
#Output files:
#
#  Data files:
#    If ITER = 0, the following files are saved, each of which contains an (optimal) parameter set for the first step fit:
#      pars_withmids_combfs_0a.sav (Best parameters for fulfilling first objective)
#      pars_withmids_combfs_1a.sav (Best parameters for fulfilling second objective)
#      pars_withmids_combfs_2a.sav (Parameters performing well in both objectives)
#      pars_withmids_combfs_wholepop.sav (All parameter sets and the corresponding objective function values)
#    If 0 < ITER <= 3, the following files are saved, each of which contains an (optimal) parameter set for the second step fit
#      (in addition to the parameters fixed during the first step, "?" referring to number 0-2 depending on the value of ITER):
#      pars_withmids_combfs_?a_0a.sav (Best parameters for fulfilling first objective)
#      pars_withmids_combfs_?a_1a.sav (Best parameters for fulfilling second objective)
#      pars_withmids_combfs_?a_2a.sav (Parameters performing well in both objectives)
#      pars_withmids_combfs_?a_wholepop.sav (All parameter sets and the corresponding objective function values)
#    If 3 < ITER <= 12, the following files are saved, each of which contains an (optimal) parameter set for the third step fit
#      (in addition to the parameters fixed during the first two steps):
#      pars_withmids_combfs_?a_?a_0a.sav (Best parameters for fulfilling first objective)
#      pars_withmids_combfs_?a_?a_1a.sav (Best parameters for fulfilling second objective)
#      pars_withmids_combfs_?a_?a_2a.sav (Best parameters for fulfilling third objective)
#      pars_withmids_combfs_?a_?a_3a.sav (Parameters performing well in first and second objectives)
#      pars_withmids_combfs_?a_?a_4a.sav (Parameters performing well in first and third objectives)
#      pars_withmids_combfs_?a_?a_5a.sav (Parameters performing well in second and third objectives)
#      pars_withmids_combfs_?a_?a_wholepop.sav (All parameter sets and the corresponding objective function values)
#    If 12 < ITER <= 66, the following files are saved, each of which contains an (optimal) parameter set for the fourth step fit
#      (in addition to the parameters fixed during the first three steps):
#      pars_withmids_combfs_?a_?a_?a_0a.sav (Best parameters for fulfilling first objective)
#      pars_withmids_combfs_?a_?a_?a_1a.sav (Best parameters for fulfilling second objective)
#      pars_withmids_combfs_?a_?a_?a_2a.sav (Best parameters for fulfilling third objective)
#      pars_withmids_combfs_?a_?a_?a_3a.sav (Parameters performing well in first and second objectives)
#      pars_withmids_combfs_?a_?a_?a_4a.sav (Parameters performing well in first and third objectives)
#      pars_withmids_combfs_?a_?a_?a_5a.sav (Parameters performing well in second and third objectives)
#      pars_withmids_combfs_?a_?a_?a_wholepop.sav (All parameter sets and the corresponding objective function values)
#
#  Images: Some EPS files illustrating the fit to the objective functions


import matplotlib
matplotlib.use('Agg')
import numpy as np
import emoo
from pylab import *
import pickle
import snmf_protocols
from neuron import h
import mytools
import time
import random

ITER = 0
myseed = 1

#Read arguments:
argv_i_myseed = 2
argv_i_ITER = 1
if len(sys.argv) > 2 and (sys.argv[1].find("mpi") > -1 or sys.argv[2].find("mpi") > -1):
  argv_i_myseed = 5
  argv_i_ITER = 4
if len(sys.argv) > argv_i_ITER:
  ITER = int(float(sys.argv[argv_i_ITER]))
if len(sys.argv) > argv_i_myseed:
  myseed = int(float(sys.argv[argv_i_myseed]))


random.seed(myseed) #random RNG
seed(myseed)        #pylab RNG
maxPerGroup = 1 # How many best ones shall we consider per objective group

# Emoo parameters:
N_samples = [1000, 1000, 1000, 2000] #size of population (for first, second, third and fourth steps)
C_samples = [2000, 2000, 2000, 4000] #population capacity
N_generations = [20, 20, 20, 10]     #number of generations
eta_m_0s = [20, 20, 20, 20]          #mutation strength parameter
eta_c_0s = [20, 20, 20, 20]          #crossover strength parameter
p_ms = [0.5, 0.5, 0.5, 0.5]          #probability of mutation


#Stimulus protocols and quantities recorded at each step:
VARIABLES = snmf_protocols.get_snmf_variables()
STIMULI = snmf_protocols.get_stimuli()
SETUP = snmf_protocols.get_setup()
STIMTYPES = snmf_protocols.get_stimulus_types()
DATASTORTYPES = snmf_protocols.get_data_storage_types()

#Target data:
unpicklefile = open('originalrun.sav', 'r')
unpickledlist = pickle.load(unpicklefile)
unpicklefile.close()
ORIGINALDATA = unpickledlist[0]
ORIGINALDISTS = [unpickledlist[1],unpickledlist[2]]
ORIGINALXS = unpickledlist[1]+[-x for x in unpickledlist[2]]

# Define the list of objectives:
OBJECTIVES = [['f0_0', 'f0_1'],
              ['f1_0', 'f1_1'],
              ['f2_0', 'f2_1', 'f2_2'],
              ['f3_0', 'f3_1', 'f3_2']]

# Originally, these were the objectives, but they are weighted by objective_group_coeffs and summed together to form the real objectives:
OBJECTIVES_SUBFUNC = [['f0_0_0_0', 'f0_1_0_0', 'f0_2_0_0', 'f0_2_1_0', 'f0_2_2_0'],
                      ['f1_0_0_0', 'f1_0_1_0', 'f1_0_2_0', 'f1_1_0_0', 'f1_1_1_0'],
                      ['f2_0_0_0', 'f2_0_1_0', 'f2_1_0_0', 'f2_1_0_1', 'f2_1_1_0', 'f2_1_1_1', 'f2_2_0_0', 'f2_2_0_1', 'f2_2_1_0', 'f2_2_1_1'],
                      ['f3_0_0_0', 'f3_0_1_0', 'f3_1_0_0', 'f3_2_0_0', 'f3_3_0_0', 'f3_3_1_0', 'f3_3_2_0']]
objective_groups = [ [ [0,1], [2,3,4] ], # Group voltage distributions together, and traces together
                     [ [0,1,2], [3,4] ], # Group voltage distributions together, and traces together
                     [ [0,1], [2,4,6,8], [3,5,7,9] ], # Group traces together, voltage distributions together, and calcium concentration distributions together
                     [ [0,1], [2,3], [4,5,6] ] ] # Group voltage traces together, numbers of spikes in long runs together, and BAC firing alone
objective_group_coeffs = [ [ [1,5], [1,1,1] ],
                              [ [1,1,1], [1,1] ],
                              [ [1,1], [1,1,1,1], [1,1,1,1] ],
                              [ [1,1], [1,1], [1,1,1] ] ]
lens_objective_groups = [ len(x) for x in objective_groups ]
groups_with_midpoints = [ [[0]], [[0],[1],[0,1]], [[0],[1],[2],[0,1],[0,2],[1,2]], [[0],[1],[2],[3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ] 
objective_groups_with_midpoints = [ groups_with_midpoints[i-1] for i in lens_objective_groups ] 
lens_objective_groups_with_midpoints = [ len(x) for x in objective_groups_with_midpoints ]
saved_until = [0]+[maxPerGroup*x for x in cumprod(lens_objective_groups_with_midpoints)] #Number of fittings to be done for each step
saved_until_cums = cumsum(saved_until)                                                   #ITERs corresponding to the first fitting tasks of each step; see comments in the beginning

nspike_restrictions = snmf_protocols.get_nspike_restrictions() 
stop_if_no_spikes = nspike_restrictions[0] #Whether to stop without checking all objectives if for some stimulus a spike was not induced (although induced in the target data)
stop_if_nspikes = nspike_restrictions[1]   #Whether to stop without checking all objectives if for some stimulus too many spikes were induced

istep = next(i for i,x in enumerate(saved_until_cums) if x >= ITER) #Which step we are performing
stepsChosen = []                                                    #Which steps must have been performed previously
divisorNow = 1
for istep2 in range(0,istep):
  stepsChosen.append(((ITER-saved_until_cums[istep-1]-1)/maxPerGroup/divisorNow%lens_objective_groups_with_midpoints[istep2]))
  divisorNow = divisorNow * lens_objective_groups_with_midpoints[istep2]
iord = (ITER-1)%maxPerGroup
print "ITER="+str(ITER)+", steps="+str(stepsChosen)

dists_apical = [] #distances of recording locations along apical dendrite will be saved here
dists_basal = []  #distances of recording locations along apical dendrite will be saved here

FIGUREIND = 0 # A global variable for determining the names of EPS files

nseg = 20            #Number of segments per compartment
nrecsperseg = 20     #Number of recording locations per compartment
if nrecsperseg == 1:
  xsperseg = [0.5]
else:
  xsperseg = [1.0*x/(nrecsperseg+1) for x in range(1,nrecsperseg+1)]

BACdt = 5.0 #ISI between apical and somatic stimulus (needed in one of the fourth step objectives)


###################### Function definitions ###############################


#initialize_model(): a function that defines the morphology, stimuli, recordings and initial values
def initialize_model():
  global dists_apical, dists_basal
  v0 = -80
  ca0 = 0.0001
  fs = 8
  tstop = 15000.0
  icell = 0

  anyRecsRecorded = 0
  for istims in range(0,len(SETUP[istep])):
    stims = SETUP[istep][istims]
    irecs = stims[2]
    if type(irecs) is not list:
      irecs = [irecs]
    if any([x > 0 for x in irecs]):
      anyRecsRecorded = 1

  h("""
load_file("stdlib.hoc")
load_file("stdrun.hoc")
objref cvode
cvode = new CVode()
cvode.active(1)
cvode.atol(0.00005)
load_file("models/fourcompartment.hoc")
objref L5PC
L5PC = new fourcompartment()
access L5PC.soma
distance()
objref st1
st1 = new IClamp(0.5)
L5PC.soma st1
objref vsoma, vdend, vdend2
vsoma = new Vector()
vdend = new Vector()
vdend2 = new Vector()
objref syn1, tvec, sl
tvec = new Vector()
sl = new List()
double siteVec[2]
siteVec[0] = 0
siteVec[1] = 0.5
L5PC.apic[0] cvode.record(&v(siteVec[1]),vdend,tvec)
L5PC.apic[0] syn1 = new epsp(siteVec[1])
syn1.imax = 0
L5PC.apic[0] nseg = """+str(nseg)+"""
L5PC.apic[1] nseg = """+str(nseg)+"""
L5PC.soma cvode.record(&v(0.5),vsoma,tvec)
L5PC.soma nseg = """+str(nseg)+"""
L5PC.dend nseg = """+str(nseg)+"""
objref vrecs_apical["""+str(2*nrecsperseg)+"""], vrecs_basal["""+str(nrecsperseg)+"""], carecs_apical["""+str(2*nrecsperseg)+"""]
tstop = """+str(tstop)+"""
cai0_ca_ion = """+str(ca0)+"""
v_init = """+str(v0)+"""
""")

  dists_apical = []
  dists_basal = []
  if anyRecsRecorded:
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[0]))
      h("L5PC.apic[0] vrecs_apical["+str(j)+"] = new Vector()")
      h("L5PC.apic[0] cvode.record(&v("+str(xsperseg[j])+"),vrecs_apical["+str(j)+"],tvec)")
      h("L5PC.apic[0] carecs_apical["+str(j)+"] = new Vector()")
      h("L5PC.apic[0] cvode.record(&cai("+str(xsperseg[j])+"),carecs_apical["+str(j)+"],tvec)")
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[1]))
      h("L5PC.apic[1] vrecs_apical["+str(nrecsperseg+j)+"] = new Vector()")
      h("L5PC.apic[1] cvode.record(&v("+str(xsperseg[j])+"),vrecs_apical["+str(nrecsperseg+j)+"],tvec)")
      h("L5PC.apic[1] carecs_apical["+str(nrecsperseg+j)+"] = new Vector()")
      h("L5PC.apic[1] cvode.record(&cai("+str(xsperseg[j])+"),carecs_apical["+str(nrecsperseg+j)+"],tvec)")
    for j in range(0,nrecsperseg):
      dists_basal.append(h.distance(xsperseg[j],sec=h.L5PC.dend))
      h("L5PC.dend vrecs_basal["+str(j)+"] = new Vector()")
      h("L5PC.dend cvode.record(&v("+str(xsperseg[j])+"),vrecs_basal["+str(j)+"],tvec)")
  else:
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[0]))
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[1]))
      dists_basal.append(h.distance(xsperseg[j],sec=h.L5PC.dend))
    print "No vrecs and carecs recorded!"

#setparams(params, istep): a function that sets the conductances of different species, axial resistances, capacitances or lengths for each compartment
#Input:
#  params: A dictionary for the parameters. See snmf_protocols for the variable names
#  istep: Which step fit is this? If fourth step (istep = 3), no additional action taken, but if istep < 3, set the conductance parameters corresponding
#    to the following steps to zero
def setparams(params, istep):
  global dists_apical, dists_basal
  
  keys = params.keys()
  #Apply the new parameter values:
  lengthChanged = False
  for ikey in range(0,len(keys)):
    key = keys[ikey]
    if key[0:2] == "L_":
      lengthChanged = True
    underscoreind = key.rfind('_')
    section = key[underscoreind+1:len(key)]
    if section == "*": # If a parameter has to be same in all sections (such as ehcn)
      h("forall "+key[0:underscoreind]+" = "+str(params[key]))
    else:
      h("L5PC."+section+" "+key[0:underscoreind]+" = "+str(params[key]))

  #Assume that if one length changed then every length changed. Change also the compartment
  #diameters such that the corresponding membrane area is conserved.
  if lengthChanged:                    
    h("L5PC.soma diam = "+str(360.132/params['L_soma']))
    h("L5PC.dend diam = "+str(2821.168/params['L_dend']))
    h("L5PC.apic[0] diam = "+str(4244.628/params['L_apic[0]']))
    h("L5PC.apic[1] diam = "+str(2442.848/params['L_apic[1]']))

    dists_apical = []
    dists_basal = []
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[0]))
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j],sec=h.L5PC.apic[1]))
    for j in range(0,nrecsperseg):
      dists_basal.append(h.distance(xsperseg[j],sec=h.L5PC.dend))
    if params['L_apic[0]'] > 620:
      h("L5PC.apic[0] syn1.loc("+str(620.0/params['L_apic[0]'])+")")
    elif params['L_apic[0]'] + params['L_apic[1]'] > 620:
      h("L5PC.apic[1] syn1.loc("+str((620.0-params['L_apic[0]'])/params['L_apic[1]'])+")")
    else:
      h("L5PC.apic[1] syn1.loc(1.0)")

  #Set those conductances to zero that will be fitted at the next step:
  for istep2 in range(istep+1,4):
    vars_zero = snmf_protocols.get_variable_params(istep2)
    for iparam in range(0,len(vars_zero)):
      if vars_zero[iparam][0]=='g' and vars_zero[iparam].find('bar') > -1:
        underscoreind = vars_zero[iparam].rfind('_')
        section = vars_zero[iparam][underscoreind+1:len(vars_zero[iparam])]
        h("L5PC."+section+" "+vars_zero[iparam][0:underscoreind]+" = 0")

#run_model(istep,saveFig,stop_if_needed): a function that runs the model using the stimuli of a certain step
#Input:
#  istep (0 to 3): Which step to run
#  saveFig: Name of the EPS file to save (empty if no need to plot the time course)
#  stop_if_needed: Boolean telling whether we can quit the function after the first stimulus condition that produces too many or too
#    few spikes. This is relevant only in the fourth step, where relatively heavy simulations are performed.
#Output:
#  myValsAllAll: A data structure containing all the data needed for determining the objective function values
def run_model(istep,saveFig="",stop_if_needed=True):
  global STIMULI, SETUP, STIMTYPES, DATASTORTYPES, dists_apical, dists_basal
  time_to_quit = False
  
  myValsAllAll = [nan]*len(SETUP[istep])
  for istims in range(0,len(SETUP[istep])):
    stims = SETUP[istep][istims]
    myValsAll = [nan]*len(stims[1])
    for iamp in range(0,len(stims[1])):
      for istim in range(0,len(stims[0])):
        stimulus = STIMULI[stims[0][istim]]
        st = STIMTYPES[stimulus[0]]
        if type(stims[1][iamp]) is list:
          myamp = stims[1][iamp][istim]
        else:
          myamp = stims[1][iamp]
        h(st[0]+"."+st[4]+" = "+str(myamp))
        for iprop in range(0,len(stimulus[1])):
          thisPropVal = stimulus[1][iprop][1]
          if stimulus[1][iprop][0] == "onset" or stimulus[1][iprop][0] == "del":
            thisPropVal = thisPropVal + istim*BACdt
          h(st[0]+"."+stimulus[1][iprop][0]+" = "+str(thisPropVal))
      h.init()
      h.run()

      irecs = stims[2]
      if type(irecs) is not list:
        irecs = [irecs]

      times=np.array(h.tvec)
      Vsoma=np.array(h.vsoma)
      if any([x > 0 for x in irecs]):
        Vdend=np.concatenate((np.array(h.vrecs_apical),np.array(h.vrecs_basal)))
        Cadend=np.array(h.carecs_apical)
      else:
        Vdend=[]
        Cadend=[]

      spikes = mytools.spike_times(times,Vsoma,-35,-45)
      if len(saveFig) > 0:
        close("all")
        f,axarr = subplots(2,2)
        axarr[0,0].plot(times,Vsoma)
        axarr[0,0].set_xlim([9990,11000])
        axarr[0,0].set_title("nSpikes="+str(len(spikes)))
        if len(Vdend) > 0:
          axarr[0,1].plot(dists_apical+[-x for x in dists_basal], [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Vdend], 'bx')
        if len(Cadend) > 0:
          axarr[1,0].plot(dists_apical, [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Cadend], 'b.')
        f.savefig(saveFig+"_step"+str(istep)+"_istims"+str(istims)+"_iamp"+str(iamp)+".eps")
      if stop_if_needed and ((stop_if_no_spikes[istep][istims][iamp] > 0 and len(spikes) == 0) or (stop_if_nspikes[istep][istims][iamp] > 0 and len(spikes) >= stop_if_nspikes[istep][istims][iamp])):
        time_to_quit = True
        break

      myVals = []
      for iirec in range(0,len(irecs)):
        irec = irecs[iirec]
        if irec == 0:
          myData = [Vsoma]
        elif irec == 1:
          myData = Vdend
        elif irec == 2:
          myData = Cadend
        else:
          print "Unknown recording type: "+str(irec)
          continue
        if DATASTORTYPES[stims[3]][0] == "fixed":
          print "ind="+str(next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1]))+", val="+str(myData[0][next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1])-1])
          myVals.append([x[next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1])-1] for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "max":
          myVals.append([max(x[next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1][0]):next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1][1])]) for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "trace" or DATASTORTYPES[stims[3]][0] == "highrestrace":
          myVals.append([mytools.interpolate_extrapolate_constant(times,x,DATASTORTYPES[stims[3]][1]) for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "highrestraceandspikes":
          myVals.append([[mytools.interpolate_extrapolate_constant(times,x,DATASTORTYPES[stims[3]][1]) for x in myData],spikes])
        elif DATASTORTYPES[stims[3]][0] == "nspikes":
          myVals.append(sum([1 for x in spikes if x >= DATASTORTYPES[stims[3]][1][0] and x < DATASTORTYPES[stims[3]][1][1]]))
        else:
          print "Unknown data storage type: "+DATASTORTYPES[stims[3]][0]
          continue
      myValsAll[iamp] = myVals[:]
      if time_to_quit:
        break
    myValsAllAll[istims] = myValsAll[:]

    for istim in range(0,len(stims[0])):
      stimulus = STIMULI[stims[0][istim]]
      st = STIMTYPES[stimulus[0]]
      h(st[0]+"."+st[4]+" = 0")
    if time_to_quit:
      break

  return myValsAllAll[:]

#distdiff(xs, fs, xs_ref, fs_ref): a function to calculate the difference of membrane potential distribution along the dendrites
#Input:
#  xs: Distances of recording locations from soma, along the dendrites of the reduced model (negative for basal, positive for apical)
#  fs: Values (e.g. maximum membrane potential during or following a stimulus) along the dendrites of the reduced model
#  xs_ref: Distances of recording locations from soma, along the dendrites of the full model (negative for basal, positive for apical)
#  fs_ref: Values (e.g. maximum membrane potential during or following a stimulus) along the dendrites of the full model. The minimal
#    and maximal values of xs_ref and fs_ref are used to scale the quantities so that both dimensions (spave vs. membrane potential)
#    are as relevant a priori
def distdiff(xs, fs, xs_ref, fs_ref):
  dist = 0
  n = 0
  dist_diff_max = max(xs_ref)-min(xs_ref)
  f_diff_max = max(fs_ref)-min(fs_ref)

  for ix in range(0,len(xs)):
    if xs[ix] > max(xs_ref) or xs[ix] < min(xs_ref):
      continue
    dists2 = [(1.0*(xs[ix]-x)/dist_diff_max)**2 + (1.0*(fs[ix]-y)/f_diff_max)**2 for x,y in zip(xs_ref, fs_ref)]
    dist = dist + sqrt(min(dists2))
    n = n+1
  return 1.0*dist/n

#highrestraceandspikesdiff(data, dataref, coeff_mV=1.0/12, coeff_ms=1.0/20): a function to calculate difference between spike trains.
#By default, a mean difference of 12mV memb. pot. is penalized as much as summed distance of 20ms between nearest spikes and further as
#much as a difference of one spike
#Input:
#  data: A list [V_m, spikes] containing the membrane potentials and spike times of the reduced model
#  dataref: A list [V_m, spikes] containing the membrane potentials and spike times of the full model
#  coeff_mV: How much is the mean difference of 1mV between target and model membrane potential penalized with respect to a difference of one spike in spike count
#  coeff_ms: How much is the summed difference of 1ms in spike timings between target and model penalized with respect to a difference of one spike in spike count
#Output:
#  (difference between numbers of spikes) + coeff_ms*(summed difference between spike timings) + coeff_mV*(mean difference between membrane potentials)
def highrestraceandspikesdiff(data, dataref, coeff_mV=1.0/12, coeff_ms=1.0/20):
  trace1 = data[0]
  spikes1 = data[1]
  traceref = dataref[0]
  spikesref = dataref[1]

  meantracediffs = [1.0*mean([abs(x-y) for x,y in zip(thistrace, thistraceref)]) for thistrace, thistraceref in zip(trace1, traceref)]
  sp_N_err = abs(len(spikesref)-len(spikes1))
  sp_t_err = 0
  if len(spikesref) > 0:
    for ispike in range(0,len(spikes1)):
      sp_t_err = sp_t_err + min([abs(spikes1[ispike] - x) for x in spikesref])
  if type(coeff_mV) is list: # Assume that if there are more than one trace lists, then coeff_mV is explicitly given with each element denoting the coefficient for a separate trace
    return sum([x*y for x,y in zip(meantracediffs, coeff_mV)]) + sp_t_err * coeff_ms + sp_N_err
  else:
    return meantracediffs[0] * coeff_mV + sp_t_err * coeff_ms + sp_N_err


#func_to_optimize(parameters,istep, saveFig=False, filename="FIGUREWITHMIDSCOMBFS",stop_if_needed=True): the function which is to be minimized.
#Calls subfunc_to_optimize with the same parameters and groups the objectives according to data in objective_groups and objective_group_coeffs
def func_to_optimize(parameters,istep, saveFig=False, filename="FIGUREWITHMIDSCOMBFS",stop_if_needed=True):
  mydict = subfunc_to_optimize(parameters,istep, saveFig, filename, stop_if_needed)
  mynewdict = {}
  for j in range(0,len(OBJECTIVES[istep])):
    myval = 0
    for k in range(0,len(objective_groups[istep][j])):
      myval = myval + objective_group_coeffs[istep][j][k]*mydict[OBJECTIVES_SUBFUNC[istep][objective_groups[istep][j][k]]]
    mynewdict['f'+str(istep)+'_'+str(j)] = myval
  return mynewdict

#subfunc_to_optimize(parameters,istep, saveFig, filename,stop_if_needed): the function which is to be minimized.
#Input:
#  parameters: A dictionary for the parameters. See snmf_protocols for the variable names
#  istep: The step to perform (0 to 3)
#  saveFig: Boolean telling whether to plot how well the parameters are fitted to the target data
#  filename: The name of the EPS file to save if any
#  stop_if_needed: Boolean telling whether we can quit the function after the first stimulus condition that produces too many or too
#    few spikes. This is relevant only in the fourth step, where relatively heavy simulations are performed.
#Output:
#  mydict: Dictionary of the (sub-)objective function values
def subfunc_to_optimize(parameters,istep, saveFig, filename, stop_if_needed):

  MAXERR = 1e8 # If stop_if_needed is true, values of objective functions corresponding to those stimuli that were skipped will be 10^8
  setparams(parameters,istep)
  global ORIGINALDATA, SETUP, dists_apical, dists_basal #Reload the values that have possibly been changed by setparams
  xs = dists_apical + [-x for x in dists_basal]
  A = run_model(istep,"",stop_if_needed)

  if saveFig:
    close("all")
    f,axarr = subplots(5,2)
    axs = [axarr[0,0],axarr[1,0],axarr[2,0],axarr[3,0],axarr[4,0],axarr[0,1],axarr[1,1],axarr[2,1],axarr[3,1],axarr[4,1]]
    for iplot in range(0,10):
      iplotx = iplot%2
      iploty = iplot/2
      axs[iplot].set_position([0.08+0.5*iplotx,0.86-0.185*iploty,0.4,0.11])
    saveInd = 0
    global FIGUREIND

  mydict = {}
  for ifun in range(0,len(SETUP[istep])):
    for iamp in range(0,len(SETUP[istep][ifun][1])):
      irecs = SETUP[istep][ifun][2]
      istor = SETUP[istep][ifun][3]
      if type(irecs) is not list:
        irecs = [irecs]

      if type(A[ifun]) is not list or (type(A[ifun][iamp]) is not list and isnan(A[ifun][iamp])):
        for iirec in range(0,len(irecs)):
          mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = MAXERR
        continue
          
      for iirec in range(0,len(irecs)):
        irec = irecs[iirec]
        if istor == 0 or istor == 1: # Maxima or steady-state-values across the dendrite(s)
          if irec == 1: #Voltage, whole dendritic tree
            mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = distdiff(xs, A[ifun][iamp][iirec], ORIGINALXS, ORIGINALDATA[istep][ifun][iamp][iirec])
          if irec == 2: #Calcium, only apical dendrite
            mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = distdiff(dists_apical, A[ifun][iamp][iirec], ORIGINALDISTS[0], ORIGINALDATA[istep][ifun][iamp][iirec])
          if saveFig:
            if irec == 1: #Voltage, whole dendritic tree
              axs[saveInd].plot(xs, A[ifun][iamp][iirec],'bx')
              axs[saveInd].plot(ORIGINALXS, ORIGINALDATA[istep][ifun][iamp][iirec],'g.')
            if irec == 2: #Calcium, only apical dendrite
              axs[saveInd].plot(dists_apical, A[ifun][iamp][iirec],'bx')
              axs[saveInd].plot(ORIGINALDISTS[0], ORIGINALDATA[istep][ifun][iamp][iirec],'g.')
            axs[saveInd].set_title('f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)+"="+str(mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)]),fontsize=7)
            if saveInd < len(axs)-1:
              saveInd = saveInd+1
        elif istor == 2 or istor == 3: # Time series
          sumval = 0
          for irecloc in range(0,len(A[ifun][iamp][iirec])): # Usually only time course of soma, but techinically allowed for dendrites as well
            sumval = sumval + sum([1.0*abs(x-y) for x,y in zip(A[ifun][iamp][iirec][irecloc], ORIGINALDATA[istep][ifun][iamp][iirec][irecloc])])
            if saveFig:
              axs[saveInd].plot(A[ifun][iamp][iirec][irecloc])
              axs[saveInd].plot(ORIGINALDATA[istep][ifun][iamp][iirec][irecloc])
          mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = sumval/250.0
          if istor == 2:
            mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = sumval*5/250.0
          if saveFig:
            axs[saveInd].set_title('f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)+"="+str(mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)]),fontsize=7)
            if saveInd < len(axs)-1:
              saveInd = saveInd+1
        elif istor == 4: # Time series with spike time precision
          mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = highrestraceandspikesdiff(A[ifun][iamp][iirec], ORIGINALDATA[istep][ifun][iamp][iirec])
          if saveFig:
            for irecloc in range(0,len(A[ifun][iamp][iirec][0])): # Usually only time course of soma, but techinically allowed for dendrites as well
              axs[saveInd].plot(A[ifun][iamp][iirec][0][irecloc])
              axs[saveInd].plot(ORIGINALDATA[istep][ifun][iamp][iirec][0][irecloc])
            axs[saveInd].set_title('f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)+"="+str(mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)])+
                                   ', nSp='+str(len(A[ifun][iamp][iirec][1]))+', nSpref='+str(ORIGINALDATA[istep][ifun][iamp][iirec][1]),fontsize=8)
            if saveInd < len(axs)-1:
              saveInd = saveInd+1
        else: # Nspikes
          mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)] = 1.0*(A[ifun][iamp][iirec]-ORIGINALDATA[istep][ifun][iamp][iirec])**2
          if saveFig:
            axs[saveInd].set_title('f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)+"="+str(mydict['f'+str(istep)+'_'+str(ifun)+'_'+str(iamp)+'_'+str(iirec)]),fontsize=7)
            if saveInd < len(axs)-1:
              saveInd = saveInd+1
  if saveFig:
    for iax in range(0,len(axs)):
      for tick in axs[iax].yaxis.get_major_ticks()+axs[iax].xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    write_params_here = [6,7,8,9]
    for j in range(0,4):
      myText = ''
      varnames = [x[0] for x in VARIABLES[j]]
      nplaced = 0
      for ivarname in range(0,len(varnames)):
        if parameters.has_key(varnames[ivarname]):
          myText = myText + varnames[ivarname] + " = " + str(parameters[varnames[ivarname]])
        else:
          underscoreind = varnames[ivarname].rfind('_')
          section = varnames[ivarname][underscoreind+1:len(varnames[ivarname])]
          h("tmpvar = 999999")
          if section == "*": # If a parameter has to be same in all sections (such as ehcn)                                                                                                               
            h("tmpvar = L5PC.soma."+varnames[ivarname][0:underscoreind])
          else:
            h("tmpvar = L5PC."+section+"."+varnames[ivarname][0:underscoreind])
          myText = myText + "No " + varnames[ivarname] + " (" + str(h.tmpvar) + ")"
        nplaced = nplaced + 1
        if ivarname < len(varnames)-1:
          myText = myText + ", "
        if nplaced > 4:
          nplaced = 0; myText = myText + '\n'
      myText = myText.replace('CaDynamics_E2','CaD');
      axs[write_params_here[j]].set_xlabel(myText,fontsize=2.2)
    if FIGUREIND==-1:
      f.savefig(filename+".eps")
    else:
      f.savefig(filename+str(istep)+"_"+str(FIGUREIND)+".eps")
    FIGUREIND = FIGUREIND+1
  return mydict


# After each generation this function is called
def checkpopulation(population, columns, gen, istep):
  picklelist = [population, columns]
  file=open(h.myfilename+'_tmp'+str(gen)+'.sav', 'w')
  pickle.dump(picklelist,file)
  file.close()
  print "Generation %d done!"%gen


########################## The main code ##################################

initialize_model()

ext = chr(ord('a')+iord)
filename = "pars_withmids_combfs"
par_names = []
par_values = []
paramdict = {}
if istep > 0:
  #Determine the name of the file to load (to get the parameters from the previous step fittings)
  for ichosen in range(0,len(stepsChosen)):
    filename = filename+"_"+str(stepsChosen[ichosen])+"a"
  filename = filename[0:-1] + ext
  unpicklefile = open(filename+".sav", 'r')
  unpickledlist = pickle.load(unpicklefile)
  unpicklefile.close()
  par_names = unpickledlist[0]
  par_values = unpickledlist[1]

  #Make the dictionary and set the parameters:
  for i in range(0,len(par_names)):
    paramdict[par_names[i]] = par_values[i]  
  setparams(paramdict, istep-1)

print "filename: "+filename

#Initialize Emoo:
FIGUREIND = 0
if type(N_samples) is list:
  myemoo = emoo.Emoo(N = N_samples[istep], C = C_samples[istep], variables = VARIABLES[istep], objectives = OBJECTIVES[istep])
  myemoo.setup(eta_m_0 = eta_m_0s[istep], eta_c_0 = eta_c_0s[istep], p_m = p_ms[istep])
else:
  myemoo = emoo.Emoo(N = N_samples, C = C_samples, variables = VARIABLES[istep], objectives = OBJECTIVES[istep])
  myemoo.setup(eta_m_0 = eta_m_0s, eta_c_0 = eta_c_0s, p_m = p_ms)
myemoo.get_objectives_error = lambda params: func_to_optimize(params,istep,False)
myemoo.checkpopulation = lambda population, columns, gen: checkpopulation(population, columns, gen, istep)
h("strdef myfilename")
h("myfilename=\""+filename+"\"")

#Run Emoo:
if type(N_samples) is list:
  myemoo.evolution(generations = N_generations[istep])
else:
  myemoo.evolution(generations = N_generations)

Results = []

#Save the best parameters for each objective (and the parameters that perform well in two distinct objectives) to a .sav file and plot the corresponding results
if myemoo.master_mode:
  params_all = myemoo.getpopulation_unnormed()

  picklelist = [N_samples, C_samples, N_generations, par_names, par_values, params_all.tolist()]
  file=open(filename+'_wholepop.sav', 'w')
  pickle.dump(picklelist,file)
  file.close()

  param_fdims = range(len(VARIABLES[istep]), len(VARIABLES[istep])+len(OBJECTIVES[istep]))
  medians = [median([y[i] for y in params_all]) for i in param_fdims]
  params_f = [[1.0*y[param_fdims[j]]/medians[j] for j in range(0,len(param_fdims))] for y in params_all]
  fvals = [[sum([y[j] for j in objective_groups_with_midpoints[istep][i]]) for i in range(0,lens_objective_groups_with_midpoints[istep])] for y in params_f]
  objs = objective_groups_with_midpoints[istep]

  for iobj_group in range(0,len(objs)):
    myord = [i[0] for i in sorted(enumerate([y[iobj_group] for y in fvals]), key=lambda x:x[1])]
    j = 1
    while j < len(myord) and len(myord) > maxPerGroup:
      if all([x==y for x,y in zip(params_all[myord[j]][0:len(VARIABLES[istep])],params_all[myord[j-1]][0:len(VARIABLES[istep])])]):
        myord.pop(j)
      else:
        j = j+1
    for isamp in range(0,maxPerGroup):
      print str(iobj_group)+"_"+str(isamp)
      ext = chr(ord('a')+isamp)
      for iparam in range(0,len(VARIABLES[istep])):
        paramdict[VARIABLES[istep][iparam][0]] = params_all[myord[isamp]][iparam]
      par_names = paramdict.keys()
      par_values = [paramdict[key] for key in par_names]
      picklelist = [par_names, par_values, N_samples, C_samples, N_generations, myord]
      file=open(filename+'_'+str(iobj_group)+ext+'.sav', 'w')
      pickle.dump(picklelist,file)
      file.close()

      FIGUREIND = -1
      func_to_optimize(paramdict,istep,True,filename+'_'+str(iobj_group)+ext)

