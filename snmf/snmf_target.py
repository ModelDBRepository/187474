#Target data generation for the simpilified neuron model fitter.
#Loads the full Hay neuron model (Hay et al. 2011 Models of Neocortical Layer 5b Pyramidal Cells Capturing a Wide Range of 
#  Dendritic and Perisomatic Active Properties), simulates the response to the stimuli defined in snmf_protocols.py,
#  and saves the quantities defined in snmf_protocols.py
#hoc-code based on the NEURON implementation of the Hay model, publicly available in https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=139653
#Tuomo Maki-Marttunen, 2015-2016
#
#Running the script:
#  python snmf_target.py
#
#Files needed:
#  snmf_protocols.py:                                                        
#    Fitting protocols (which quantities are fitted at each step; what stimuli are for each objective function; etc.)
#  mytools.py:                                                                                                       
#    General tools for e.g. spike detection                                                                          
#                                                                                                                      
#Output files:      
#  originalrun.sav  
#    Target data (generated from the full model based on snmf_protocols.py)  

from neuron import h
import matplotlib
matplotlib.use('Agg')
import numpy
from pylab import *
import mytools
import pickle
import time
import sys
import random
import snmf_protocols

dists_apical = []
dists_basal = []
BACdt = 5.0

def initialize():
  global dists_apical, dists_basal, BACdt
  v0 = -80
  ca0 = 0.0001
  BACdt = 5.0
  fs = 8
  tstop = 15000.0
  icell = 0
  nsegs = 10
  nrecsperseg = 5

  if nrecsperseg == 1:
    xsperseg = [0.5]
  else:
    xsperseg = [1.0*x/(nrecsperseg+1) for x in range(1,nrecsperseg+1)]

  h("""
load_file("stdlib.hoc")
load_file("stdrun.hoc")
objref cvode
cvode = new CVode()
cvode.active(1)
cvode.atol(0.00005)
load_file("import3d.hoc")
objref L5PC              
load_file(\"models/L5PCbiophys3.hoc\")
load_file(\"models/L5PCtemplate.hoc\")
L5PC = new L5PCtemplate(\"morphologies/cell1.asc\")
access L5PC.soma
objref st1
objref vsoma, vdend, vdend2
vsoma = new Vector()
vdend = new Vector()
vdend2 = new Vector()
objref syn1, tvec, sl
tvec = new Vector()
sl = new List()
double siteVec[2]
sl = L5PC.locateSites("apic",620)       
maxdiam = 0                             
for(i=0;i<sl.count();i+=1){             
  dd1 = sl.o[i].x[1]                    
  dd = L5PC.apic[sl.o[i].x[0]].diam(dd1)
  if (dd > maxdiam) {                   
    j = i                               
    maxdiam = dd                        
  }                                     
}                                       
siteVec[0] = sl.o[j].x[0]               
siteVec[1] = sl.o[j].x[1]
L5PC.apic[siteVec[0]] cvode.record(&v(siteVec[1]),vdend,tvec)
L5PC.apic[siteVec[0]] syn1 = new epsp(siteVec[1])
sl = L5PC.locateSites("apic",800)       
maxdiam = 0                             
for(i=0;i<sl.count();i+=1){             
  dd1 = sl.o[i].x[1]                    
  dd = L5PC.apic[sl.o[i].x[0]].diam(dd1)
  if (dd > maxdiam) {                   
    j = i                               
    maxdiam = dd                        
  }                                     
}                                       
siteVec[0] = sl.o[j].x[0]               
siteVec[1] = sl.o[j].x[1]
L5PC.apic[siteVec[0]] cvode.record(&v(siteVec[1]),vdend2,tvec)
syn1.imax = 0
L5PC.soma cvode.record(&v(0.5),vsoma,tvec)
forall nseg = """+str(nsegs)+"""
tstop = """+str(tstop)+"""

objref vrecs_apical["""+str(nrecsperseg*109)+"""], vrecs_basal["""+str(nrecsperseg*84)+"""], carecs_apical["""+str(nrecsperseg*109)+"""]

cai0_ca_ion = """+str(ca0)+"""
v_init = """+str(v0)+"""
""")

  dists_apical = []
  dists_basal = []
  for i in range(0,109):
    h("access L5PC.apic["+str(i)+"]")
    for j in range(0,nrecsperseg):
      dists_apical.append(h.distance(xsperseg[j]))
      h("L5PC.apic["+str(i)+"] vrecs_apical["+str(i*nrecsperseg + j)+"] = new Vector()")
      h("L5PC.apic["+str(i)+"] carecs_apical["+str(i*nrecsperseg + j)+"] = new Vector()")
      h("L5PC.apic["+str(i)+"] cvode.record(&v("+str(xsperseg[j])+"),vrecs_apical["+str(i*nrecsperseg + j)+"],tvec)")
      h("L5PC.apic["+str(i)+"] cvode.record(&cai("+str(xsperseg[j])+"),carecs_apical["+str(i*nrecsperseg + j)+"],tvec)")
  for i in range(0,84):
    h("access L5PC.dend["+str(i)+"]")
    for j in range(0,nrecsperseg):
      dists_basal.append(h.distance(xsperseg[j]))
      h("L5PC.dend["+str(i)+"] vrecs_basal["+str(i*nrecsperseg + j)+"] = new Vector()")
      h("L5PC.dend["+str(i)+"] cvode.record(&v("+str(xsperseg[j])+"),vrecs_basal["+str(i*nrecsperseg + j)+"],tvec)")
  h("access L5PC.soma")
  maxLenApic = 1300
  maxLenBasal = 280


  STIMTYPES = snmf_protocols.get_stimulus_types()
  for istim_type in range(0,len(STIMTYPES)):
    ST = STIMTYPES[istim_type]
    if ST[3][0] == "fixed":
      segname = "L5PC."+ST[2]
      segx = ST[3][1]
    elif ST[3][0] == "distance":
      h("""
sl = L5PC.locateSites(\""""+ST[2]+"""\","""+str(ST[3][1])+""")
maxdiam = 0                             
for(i=0;i<sl.count();i+=1){             
  dd1 = sl.o[i].x[1]                    
  dd = L5PC."""+ST[2]+"""[sl.o[i].x[0]].diam(dd1)
  if (dd > maxdiam) {                   
    j = i                               
    maxdiam = dd                        
  }                                     
}                                       
siteVec[0] = sl.o[j].x[0]               
siteVec[1] = sl.o[j].x[1]
""")
      segname = "L5PC."+ST[2]+"["+str(int(h.siteVec[0]))+"]"
      segx = h.siteVec[1]
    else:
      print "Error: use \"fixed\" or \"distance\" for ST[3][0]!"
    print "istim_type = "+str(istim_type)
    print("""
objref """+ST[0]+"""
""" +segname + """ """ + ST[0] + """ = new """+ST[1]+"""("""+str(segx)+""")
""") 
    h("""
objref """+ST[0]+"""
""" +segname + """ """ + ST[0] + """ = new """+ST[1]+"""("""+str(segx)+""")
""")

  return [dists_apical, dists_basal]



def run(saveFig=False):

  STIMULI = snmf_protocols.get_stimuli()
  SETUP = snmf_protocols.get_setup()

  approx_to_real_seclist = {"soma": "somatic", "dend": "basal", "apic[0]": "apical", "apic[1]": "apical"}

  STIMTYPES = snmf_protocols.get_stimulus_types()
  DATASTORTYPES = snmf_protocols.get_data_storage_types()

  myReturn = [[]]*len(SETUP)
  for istep in range(len(SETUP)-1,-1,-1):
    vars_zero = snmf_protocols.get_variable_params(istep+1)
    for iparam in range(0,len(vars_zero)):
      if vars_zero[iparam][0]=='g' and vars_zero[iparam].find('bar') > -1:
        print vars_zero[iparam]
        underscoreind = vars_zero[iparam].rfind('_')
        section = vars_zero[iparam][underscoreind+1:len(vars_zero[iparam])]
        print "forsec L5PC."+approx_to_real_seclist[section]+" "+vars_zero[iparam][0:underscoreind]+" = 0" 
        h("forsec L5PC."+approx_to_real_seclist[section]+" "+vars_zero[iparam][0:underscoreind]+" = 0")

    myValsAllAll = []
    for istims in range(0,len(SETUP[istep])):
      stims = SETUP[istep][istims]
      myValsAll = []
      for iamp in range(0,len(stims[1])):
        for istim in range(0,len(stims[0])):
          stimulus = STIMULI[stims[0][istim]]
          st = STIMTYPES[stimulus[0]]          
          if type(stims[1][iamp]) is list:
            myamp = stims[1][iamp][istim]
          else:
            myamp = stims[1][iamp]
          print(st[0]+"."+st[4]+" = "+str(myamp))
          h(st[0]+"."+st[4]+" = "+str(myamp))
          for iprop in range(0,len(stimulus[1])):
            thisPropVal = stimulus[1][iprop][1]
            if stimulus[1][iprop][0] == "onset" or stimulus[1][iprop][0] == "del":
              thisPropVal = thisPropVal + istim*BACdt
            print st[0]+"."+stimulus[1][iprop][0]+" = "+str(thisPropVal)
            h(st[0]+"."+stimulus[1][iprop][0]+" = "+str(thisPropVal))
        h.init()
        timenow = time.time()
        h.run()
        print "Step "+str(istep)+", istims = "+str(istims)+", iamp = "+str(iamp)+" completed in "+str(time.time()-timenow)+" seconds"
        times=np.array(h.tvec)
        Vsoma=np.array(h.vsoma)
        Vdend=np.concatenate((np.array(h.vrecs_apical),np.array(h.vrecs_basal)))
        Cadend=np.array(h.carecs_apical)
        spikes = mytools.spike_times(times,Vsoma,-35,-45)
        if saveFig:
          close("all")
          f,axarr = subplots(2,2)
          axarr[0,0].plot(times,Vsoma)
          axarr[0,0].set_xlim([9990,11000])
          axarr[0,0].set_title("nSpikes="+str(len(spikes)))
          axarr[0,1].plot(dists_apical+[-x for x in dists_basal], [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Vdend], 'bx')
          axarr[1,0].plot(dists_apical, [max([x[i] for i,t in enumerate(times) if t >= 9000]) for x in Cadend], 'b.')
          f.savefig("original_step"+str(istep)+"_istims"+str(istims)+"_iamp"+str(iamp)+".eps")
        #times = range(0,15000); Vsoma = range(0,15000); Vdend = [range(0,15000), range(0,15000)]; Cadend = Vdend; spikes = []
        print "istep="+str(istep)+", istims="+str(istims)+", iamp="+str(iamp)+": nSpikes="+str(len(spikes))
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
          print "istep="+str(istep)+": "
          print DATASTORTYPES[stims[3]]
          try:
            print "min(myData)="+str(min(myData))+", max(myData)="+str(max(myData))
          except:
            try:
              print "min(myData[0])="+str(min(myData[0]))+", max(myData[0])="+str(max(myData[0]))
            except:
              print myData
          if DATASTORTYPES[stims[3]][0] == "fixed":
            print str(next(i for i,t in enumerate(times) if t >= DATASTORTYPES[stims[3]][1]))
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
        myValsAll.append(myVals[:])
      myValsAllAll.append(myValsAll[:])
          
      for istim in range(0,len(stims[0])):
        stimulus = STIMULI[stims[0][istim]]
        st = STIMTYPES[stimulus[0]]          
        print(st[0]+"."+st[4]+" = 0")
        h(st[0]+"."+st[4]+" = 0")

    myReturn[istep] = myValsAllAll[:]
  picklelist = [myReturn, dists_apical, dists_basal]
  file = open('originalrun.sav', 'w')
  pickle.dump(picklelist,file)
  file.close()
  #print "run():"
  #mytools.printlistlen(myReturn)
  return myReturn


#calculate the error from mean experimental values (measured in terms of SDs, except #APs, which are absolute)
def howfarfrommeans():
  A = run()
  errors = [mean([abs(A[0][0]-9.0)/0.88,      abs(A[0][1]-14.5)/0.56,     abs(A[0][2]-22.5)/2.22]),
            mean([abs(A[1][0]-0.0036)/0.0091, abs(A[1][1]-0.0023)/0.0056, abs(A[1][2]-0.0046)/0.0026]),
            mean([abs(A[2][0]-0.1204)/0.0321, abs(A[2][1]-0.1086)/0.0368, abs(A[2][2]-0.0954)/0.0140]),
            mean([abs(A[3][0]-57.75)/33.48,   abs(A[3][1]-6.625)/8.65,    abs(A[3][2]-5.38)/0.83]),
            mean([abs(A[4][0]-43.25)/7.32,    abs(A[4][1]-19.13)/7.31,    abs(A[4][2]-7.25)/1.0]),
            mean([abs(A[5][0]-26.23)/4.97,    abs(A[5][1]-16.52)/6.11,    abs(A[5][2]-16.44)/6.93]),
            mean([abs(A[6][0]+58.04)/4.58,    abs(A[6][1]+60.51)/4.67,    abs(A[6][2]+59.99)/3.92]),
            mean([abs(A[7][0]-0.238)/0.03,    abs(A[7][1]-0.279)/0.027,   abs(A[7][2]-0.213)/0.037]),
            mean([abs(A[8][0]-1.31)/0.17,     abs(A[8][1]-1.38)/0.28,     abs(A[8][2]-1.86)/0.41]),
            abs(A[9]-6.73)/2.54,
            abs(A[10]-37.43)/1.27,
            abs(A[11]-3.0),
            abs(A[12]-9.9)/0.85,
            abs(A[13]+65.0)/4.0,
            abs(A[14]-25.0)/5.0,
            abs(A[15]-2.0)/0.5,
            abs(A[16]-1.0)]
  return errors
            
  
initialize()
run(True)
