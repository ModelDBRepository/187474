#drawfits_withmids_combfs
#A script that illustrates how well a certain parameter set fits to the target data and calculates the objective functions
#Tuomo Maki-Marttunen, 2015-2016
#
#Running the script:
#  python drawfits_withmids_combfs.py filename
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
#  step0_<<filename>>_nseg5.eps: Figure with step 1 objectives (active conductances blocked)
#  step1_<<filename>>_nseg5.eps: Figure with step 2 objectives (active conductances except for HCN blocked)
#  step2_<<filename>>_nseg5.eps: Figure with step 3 objectives (voltage-gated sodium and potassium channels blocked)
#  step3_<<filename>>_nseg5.eps: Figure with step 4 objectives (intact neuron)
#

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

def strminNdecimals(x, N=2):
  if abs(x) < 1:
    return ("%."+str(N)+"g") % x
  return ("%."+str(N)+"f") % x
  
filename = 'pars_withmids_combfs_final'

if len(sys.argv) > 1: 
  filename = sys.argv[1]

VARIABLES = snmf_protocols.get_snmf_variables()

STIMULI = snmf_protocols.get_stimuli()
SETUP = snmf_protocols.get_setup()
STIMTYPES = snmf_protocols.get_stimulus_types()
DATASTORTYPES = snmf_protocols.get_data_storage_types()

unpicklefile = open('originalrun.sav', 'r')
unpickledlist = pickle.load(unpicklefile)
unpicklefile.close()
ORIGINALDATA = unpickledlist[0]
ORIGINALDISTS = [unpickledlist[1],unpickledlist[2]]
ORIGINALXS = unpickledlist[1]+[-x for x in unpickledlist[2]]



# Originally, these were the objectives, but they are weighted by objective_group_coeffs and summed together to form the real objectives.
OBJECTIVES_SUBFUNC = [['f0_0_0_0', 'f0_1_0_0', 'f0_2_0_0', 'f0_2_1_0', 'f0_2_2_0'],
                      ['f1_0_0_0', 'f1_0_1_0', 'f1_0_2_0', 'f1_1_0_0', 'f1_1_1_0'],
                      ['f2_0_0_0', 'f2_0_1_0', 'f2_1_0_0', 'f2_1_0_1', 'f2_1_1_0', 'f2_1_1_1', 'f2_2_0_0', 'f2_2_0_1', 'f2_2_1_0', 'f2_2_1_1'],
                      ['f3_0_0_0', 'f3_0_1_0', 'f3_1_0_0', 'f3_2_0_0', 'f3_3_0_0', 'f3_3_1_0', 'f3_3_2_0']]
# Define the list of objectives            
OBJECTIVES = [['f0_0', 'f0_1'],
              ['f1_0', 'f1_1'],
              ['f2_0', 'f2_1', 'f2_2'],
              ['f3_0', 'f3_1', 'f3_2']]
objective_groups = [ [ [0,1], [2,3,4] ], # Group voltage distributions together, and traces together                                                           
                     [ [0,1,2], [3,4] ], # Group voltage distributions together, and traces together                                                           
                     [ [0,1], [2,4,6,8], [3,5,7,9] ], # Group traces together, voltage distributions together, and calcium concentration distributions together
                     [ [0,1], [2,3], [4,5,6] ] ] # Group voltage traces together, numbers of spikes in long runs together, and BAC firing alone                
objective_group_coeffs = [ [ [1,5], [1,1,1] ],
                              [ [1,1,1], [1,1] ],
                              [ [1,1], [1,1,1,1], [1,1,1,1] ],
                              [ [1,1], [1,1], [1,1,1] ] ]
OBJECTIVES_SUBFUNC_ISPLOTTED = [[1,1,1,1,1],
                                [1,1,1,1,1],
                                [1,1,1,1,1,1,1,1,1,1],
                                [1,1,1,1,1,0,0]]
OBJECTIVES_SUBFUNC_TITLES = [["1.1","1.2","1.3a","1.3b","1.3c"],
                             ["2.1a","2.1b","2.1c","2.2a","2.2b"],
                             ["3.1a","3.1b","3.2a","3.3a","3.2b","3.3b","3.4a","3.5a","3.4b","3.5b"],
                             ["4.1a","4.1b","4.2","4.3","4.4a","4.4b","4.4c"]]
OBJECTIVE_LABELS = ["x (um)", "t (ms)", "I (nA)", "V_m (mV)", "[Ca2+] (mM)", "f (Hz)"]
OBJECTIVE_XLABS = [[0,0,1,1,1],[0,0,0,1,1],[1,1,0,0,0,0,0,0,0,0],[1,1,1,1,2]]
OBJECTIVE_YLABS = [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,4,3,4,3,4,3,4],[3,3,3,3,5]]

AMP_LABELS = ["I = ", "I_max = ", "[I, I_max] = "]
AMP_LABS = [[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0,0,1,1,1,1],[0,0,0,2,0]]
AMP_POSS = [[1,1,2,0,0],[0,1,1,1,1],[1,1,1,1,1,1,0,0,0,0],[1,1,1,1,-1]]
YLIMS = [[[],[],[-170,-80],[-130,-40],[-100,-10]],[[],[],[],[],[]],[[],[],[],[],[],[],[],[],[],[]],[[],[],[],[],[]]]

# Define the list of objectives
OBJECTIVES = [['f0_0', 'f0_1'],
              ['f1_0', 'f1_1'],
              ['f2_0', 'f2_1', 'f2_2'],
              ['f3_0', 'f3_1','f3_2', 'f3_3']]

dists_apical = []
dists_basal = []

nseg = 5
nrecsperseg = 5
if nrecsperseg == 1:
  xsperseg = [0.5]
else:
  xsperseg = [1.0*x/(nrecsperseg+1) for x in range(1,nrecsperseg+1)]
BACdt = 5.0

def boxoff(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def initialize_model():
  global dists_apical, dists_basal
  v0 = -80
  ca0 = 0.0001
  fs = 8
  tstop = 15000.0
  icell = 0

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
      print "forall "+key[0:underscoreind]+" = "+str(params[key])
      h("forall "+key[0:underscoreind]+" = "+str(params[key]))
    else:
      print "L5PC."+section+" "+key[0:underscoreind]+" = "+str(params[key])
      h("L5PC."+section+" "+key[0:underscoreind]+" = "+str(params[key]))

  if lengthChanged: #Assume that if one length changed then every length changed
    print "L5PC.soma diam = "+str(360.132/params['L_soma'])
    print "L5PC.dend diam = "+str(2821.168/params['L_dend'])
    print "L5PC.apic[0] diam = "+str(4244.628/params['L_apic[0]'])
    print "L5PC.apic[1] diam = "+str(2442.848/params['L_apic[1]'])
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
      #print "L5PC.apic[0] syn1.loc("+str(620.0/params['L_apic[0]'])+")"
      h("L5PC.apic[0] syn1.loc("+str(620.0/params['L_apic[0]'])+")")
    elif params['L_apic[0]'] + params['L_apic[1]'] > 620:
      #print "L5PC.apic[1] syn1.loc("+str((620.0-params['L_apic[0]'])/params['L_apic[1]'])+")"
      h("L5PC.apic[1] syn1.loc("+str((620.0-params['L_apic[0]'])/params['L_apic[1]'])+")")
    else:
      #print "L5PC.apic[1] syn1.loc(1.0)"
      h("L5PC.apic[1] syn1.loc(1.0)")


  #Set those conductances to zero that will be fitted at the next step:
  for istep2 in range(istep+1,4):
    vars_zero = snmf_protocols.get_variable_params(istep2)
    for iparam in range(0,len(vars_zero)):
      if vars_zero[iparam][0]=='g' and vars_zero[iparam].find('bar') > -1:
        #print vars_zero[iparam]
        underscoreind = vars_zero[iparam].rfind('_')
        section = vars_zero[iparam][underscoreind+1:len(vars_zero[iparam])]
        print "L5PC."+section+" "+vars_zero[iparam][0:underscoreind]+" = 0"
        h("L5PC."+section+" "+vars_zero[iparam][0:underscoreind]+" = 0")


def run_model(istep,saveFig="",stop_if_needed=True):
  global STIMULI, SETUP, STIMTYPES, DATASTORTYPES, dists_apical, dists_basal
  
  myValsAllAll = [nan]*len(SETUP[istep])
  for istims in range(0,len(SETUP[istep])):
    stims = SETUP[istep][istims]
    #print "stims:"
    #print stims
    myValsAll = [nan]*len(stims[1])
    for iamp in range(0,len(stims[1])):
      for istim in range(0,len(stims[0])):
        stimulus = STIMULI[stims[0][istim]]
        st = STIMTYPES[stimulus[0]]
        if type(stims[1][iamp]) is list:
          myamp = stims[1][iamp][istim]
        else:
          myamp = stims[1][iamp]
        #print(st[0]+"."+st[4]+" = "+str(myamp))
        h(st[0]+"."+st[4]+" = "+str(myamp))
        #print "len(stimulus[1]) = "+str(len(stimulus[1]))
        for iprop in range(0,len(stimulus[1])):
          thisPropVal = stimulus[1][iprop][1]
          if stimulus[1][iprop][0] == "onset" or stimulus[1][iprop][0] == "del":
            #print "stimulus[1]:"
            #print stimulus[1]
            thisPropVal = thisPropVal + istim*BACdt
            #print "iprop="+str(iprop)+", istim="+str(istim)
          #print st[0]+"."+stimulus[1][iprop][0]+" = "+str(thisPropVal)
          h(st[0]+"."+stimulus[1][iprop][0]+" = "+str(thisPropVal))
      h.init()
      timenow = time.time()
      print "st1.amp="+str(h.st1.amp)+", st1.delay="+str(h.st1.delay)+", st1.dur="+str(h.st1.dur)+", syn1.imax="+str(h.syn1.imax)+", syn1.onset="+str(h.syn1.onset)+", syn1.tau0="+str(h.syn1.tau0)+", syn1.tau1="+str(h.syn1.tau1)
      h.run()
      print "Step "+str(istep)+", istims = "+str(istims)+", iamp = "+str(iamp)+" completed in "+str(time.time()-timenow)+" seconds"
      times=np.array(h.tvec)
      Vsoma=np.array(h.vsoma)
      Vdend=np.concatenate((np.array(h.vrecs_apical),np.array(h.vrecs_basal)))
      Cadend=np.array(h.carecs_apical)
      spikes = mytools.spike_times(times,Vsoma,-35,-45)
      if len(saveFig) > 0:
        close("all")
        f,axarr = subplots(2,2)
        axarr[0,0].plot(times,Vsoma)
        axarr[0,0].set_xlim([9990,11000])
        axarr[0,0].set_title("nSpikes="+str(len(spikes)))
        axarr[0,1].plot(dists_apical+[-x for x in dists_basal], [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Vdend], 'bx')
        axarr[1,0].plot(dists_apical, [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Cadend], 'b.')
        f.savefig(saveFig+"_step"+str(istep)+"_istims"+str(istims)+"_iamp"+str(iamp)+".eps")

      irecs = stims[2]
      if type(irecs) is not list:
        irecs = [irecs]
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
        #print myData
        #print DATASTORTYPES[stims[3]]
        #print str(len(myData))
        if DATASTORTYPES[stims[3]][0] == "fixed":
          print "ind="+str(next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1]))+", val="+str(myData[0][next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1])-1])
          myVals.append([x[next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1])-1] for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "max":
          myVals.append([max(x[next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1][0]):next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1][1])]) for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "trace" or DATASTORTYPES[stims[3]][0] == "highrestrace":
          myVals.append([mytools.interpolate(times,x,DATASTORTYPES[stims[3]][1]) for x in myData])
        elif DATASTORTYPES[stims[3]][0] == "highrestraceandspikes":
          myVals.append([[mytools.interpolate(times,x,DATASTORTYPES[stims[3]][1]) for x in myData],spikes])
        elif DATASTORTYPES[stims[3]][0] == "nspikes":
          myVals.append(sum([1 for x in spikes if x >= DATASTORTYPES[stims[3]][1][0] and x < DATASTORTYPES[stims[3]][1][1]]))
        else:
          print "Unknown data storage type: "+DATASTORTYPES[stims[3]][0]
          continue
      #print "myVals:"+str(myVals)
      myValsAll[iamp] = myVals[:]
    #print "myValsAll:"+str(myValsAll)
    myValsAllAll[istims] = myValsAll[:]

    for istim in range(0,len(stims[0])):
      stimulus = STIMULI[stims[0][istim]]
      st = STIMTYPES[stimulus[0]]
      #print(st[0]+"."+st[4]+" = 0")
      h(st[0]+"."+st[4]+" = 0")

  #print "run():"
  #mytools.printlistlen(myValsAllAll[:])
  return myValsAllAll[:]


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

def highrestraceandspikesdiff(data, dataref, coeff_mV=1.0/12, coeff_ms=1.0/20): #By default, a mean difference of 12mV memb. pot. <=> summed distance of 20ms between nearest spikes <=> a difference of one spike,
  trace1 = data[0]
  spikes1 = data[1]
  traceref = dataref[0]
  spikesref = dataref[1]
  #print str(trace1)
  #print str(spikes1)

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


# This is the function which is going to be minimized
def drawstuff(parameters,istep, filename):
  MAXERR = 1e8
  setparams(parameters,istep)
  global ORIGINALDATA, SETUP, dists_apical, dists_basal #Reload the values that have possibly been changed by setparams
  xs = dists_apical + [-x for x in dists_basal]
  A = run_model(istep,"",False)

  close("all")
  nfig = sum(OBJECTIVES_SUBFUNC_ISPLOTTED[istep])
  ny = (nfig+1)/2
  f,axs = subplots(nfig,1)
  for iplot in range(0,nfig):
    iplotx = iplot%2
    iploty = iplot/2
    axs[iplot].set_position([0.08+0.5*iplotx,0.1+(ny-iploty-1)*(0.85/ny),0.4,0.85/ny-0.08])
    axs[iplot].locator_params(axis='y',nbins=6)
    axs[iplot].get_yaxis().get_major_formatter().set_useOffset(False)
    axs[iplot].get_yaxis().get_major_formatter().set_scientific(False)
  saveInd = 0


  IF_nspikes = []
  IF_nspikes_control = []
  IF_amps = [0.78, 1.0, 1.9]
  IF_saveInd = nan
  iobj = -1
  for ifun in range(0,len(SETUP[istep])):
    for iamp in range(0,len(SETUP[istep][ifun][1])):
      iobj = iobj+1
      if not OBJECTIVES_SUBFUNC_ISPLOTTED[istep][iobj]:
        continue
      
      irecs = SETUP[istep][ifun][2]
      istor = SETUP[istep][ifun][3]
      if type(irecs) is not list:
        irecs = [irecs]

      for iirec in range(0,len(irecs)):
        irec = irecs[iirec]
        if istor == 0 or istor == 1: # Maxima or steady-state-values across the dendrite(s)
          if irec == 1: #Voltage, whole dendritic tree
            thiserr = strminNdecimals(distdiff(xs, A[ifun][iamp][iirec], ORIGINALXS, ORIGINALDATA[istep][ifun][iamp][iirec]))
            axs[saveInd].plot(xs, A[ifun][iamp][iirec],'bx')
            axs[saveInd].plot(ORIGINALXS, ORIGINALDATA[istep][ifun][iamp][iirec],'g.')
          if irec == 2: #Calcium, only apical dendrite
            thiserr = strminNdecimals(distdiff(dists_apical, A[ifun][iamp][iirec], ORIGINALDISTS[0], ORIGINALDATA[istep][ifun][iamp][iirec]))
            axs[saveInd].plot(dists_apical, A[ifun][iamp][iirec],'bx')
            axs[saveInd].plot(ORIGINALDISTS[0], ORIGINALDATA[istep][ifun][iamp][iirec],'g.')
        elif istor == 2 or istor == 3: # Time series
          sumval = 0
          myts = [5*x for x in range(0,51)]
          if istor==3:
            myts = range(0,251)
          for irecloc in range(0,len(A[ifun][iamp][iirec])): # Usually only time course of soma, but techinically allowed for dendrites as well
            sumval = sumval + mean([1.0*abs(x-y) for x,y in zip(A[ifun][iamp][iirec][irecloc], ORIGINALDATA[istep][ifun][iamp][iirec][irecloc])])
            axs[saveInd].plot(myts,A[ifun][iamp][iirec][irecloc])
            axs[saveInd].plot(myts,ORIGINALDATA[istep][ifun][iamp][iirec][irecloc])
          thiserr = strminNdecimals(sumval)
        elif istor == 4: # Time series with spike time precision
          thiserr = strminNdecimals(highrestraceandspikesdiff(A[ifun][iamp][iirec], ORIGINALDATA[istep][ifun][iamp][iirec]))
          for irecloc in range(0,len(A[ifun][iamp][iirec][0])): # Usually only time course of soma, but techinically allowed for dendrites as well
            axs[saveInd].plot(A[ifun][iamp][iirec][0][irecloc])
            axs[saveInd].plot(ORIGINALDATA[istep][ifun][iamp][iirec][0][irecloc])
        else: # Nspikes
          IF_nspikes.append(A[ifun][iamp][iirec])
          IF_nspikes_control.append(ORIGINALDATA[istep][ifun][iamp][iirec])
          IF_nspikes.append(A[ifun][iamp+1][iirec])
          IF_nspikes_control.append(ORIGINALDATA[istep][ifun][iamp+1][iirec])
          IF_nspikes.append(A[ifun][iamp+2][iirec])
          IF_nspikes_control.append(ORIGINALDATA[istep][ifun][iamp+2][iirec])

          IF_saveInd = saveInd
          thiserr = strminNdecimals(1.0*(A[ifun][iamp][iirec]-ORIGINALDATA[istep][ifun][iamp][iirec])**2)
        axs[saveInd].set_title('f'+OBJECTIVES_SUBFUNC_TITLES[istep][saveInd]+" = "+thiserr,fontsize=7)
        print "saveInd = "+str(saveInd)+", iobj="+str(iobj)+", OBJECTIVE_LABELS[OBJECTIVE_XLABS[istep][saveInd]]="+str(OBJECTIVE_LABELS[OBJECTIVE_XLABS[istep][saveInd]])+', f'+OBJECTIVES_SUBFUNC_TITLES[istep][saveInd]+', thiserr='+str(thiserr)
        axs[saveInd].set_xlabel(OBJECTIVE_LABELS[OBJECTIVE_XLABS[istep][saveInd]],fontsize=7)
        axs[saveInd].set_ylabel(OBJECTIVE_LABELS[OBJECTIVE_YLABS[istep][saveInd]],fontsize=7)
        if len(YLIMS[istep][saveInd]) > 0:
          axs[saveInd].set_ylim(YLIMS[istep][saveInd])
        ax = axs[saveInd].axis()
        if AMP_POSS[istep][saveInd] == 0:
          axs[saveInd].text(ax[0]+(ax[1]-ax[0])*0.05,                                 ax[2]+(ax[3]-ax[2])*0.75,AMP_LABELS[AMP_LABS[istep][saveInd]]+str(SETUP[istep][ifun][1][iamp])+" nA")
        elif AMP_POSS[istep][saveInd] == 1:
          axs[saveInd].text(ax[0]+(ax[1]-ax[0])*(0.7-0.1*AMP_LABS[istep][saveInd]**2),ax[2]+(ax[3]-ax[2])*0.75,AMP_LABELS[AMP_LABS[istep][saveInd]]+str(SETUP[istep][ifun][1][iamp])+" nA")
        elif AMP_POSS[istep][saveInd] == 2:
          axs[saveInd].text(ax[0]+(ax[1]-ax[0])*0.05,                                 ax[2]+(ax[3]-ax[2])*0.25,AMP_LABELS[AMP_LABS[istep][saveInd]]+str(SETUP[istep][ifun][1][iamp])+" nA")
        elif AMP_POSS[istep][saveInd] == 3:
          axs[saveInd].text(ax[0]+(ax[1]-ax[0])*(0.7-0.1*AMP_LABS[istep][saveInd]**2),ax[2]+(ax[3]-ax[2])*0.25,AMP_LABELS[AMP_LABS[istep][saveInd]]+str(SETUP[istep][ifun][1][iamp])+" nA")
          
        saveInd = saveInd+1
  if not isnan(IF_saveInd):
    print IF_amps
    print IF_nspikes
    print IF_nspikes_control
    axs[IF_saveInd].plot(IF_amps,[x/3.0 for x in IF_nspikes],'bx-')
    axs[IF_saveInd].plot(IF_amps,[x/3.0 for x in IF_nspikes_control],'g.-')
    #axs[IF_saveInd].set_title('f'+OBJECTIVES_SUBFUNC_TITLES[istep][saveInd][:-1]+" = "+thiserr,fontsize=7)
    axs[IF_saveInd].set_title('f'+OBJECTIVES_SUBFUNC_TITLES[istep][saveInd][:-1]+" = "+str(1.0*(A[ifun][iamp][iirec]-ORIGINALDATA[istep][ifun][iamp][iirec])**2),fontsize=7)

  for iax in range(0,len(axs)):
    for tick in axs[iax].yaxis.get_major_ticks()+axs[iax].xaxis.get_major_ticks():
      tick.label.set_fontsize(5)

  if istep == 0:
    ax = axs[0].axis()
    axs[0].text(ax[0]-0.2*(ax[1]-ax[0]),ax[2]+0.9*(ax[3]-ax[2]),'B',fontsize=30)

  f.savefig(filename+".eps")

paramdict = {}

unpicklefile = open(filename+".sav", 'r')
unpickledlist = pickle.load(unpicklefile)
unpicklefile.close()
par_names = unpickledlist[0]
par_values = unpickledlist[1]
for i in range(0,len(par_names)):
  paramdict[par_names[i]] = par_values[i]

initialize_model()

for istep in range(0,4):  
  FIGUREIND = -1
  drawstuff(paramdict,istep,"step"+str(istep)+"_"+filename+"_nseg5")

