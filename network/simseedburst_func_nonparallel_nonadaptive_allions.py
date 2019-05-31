# Library for running non-parallel network simulation of reduced-morphology L5PCs using small (200 Exc. & 50 Inh.) number of synapses.
# The HOC-code is based on the network model by Hay & Segev 2015: Dendritic Excitability and Gain Control in Recurrent Cortical Microcircuits,
#   publicly available at https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=156780
#
# Tuomo Maki-Marttunen, 2015-2016
#
# Files needed:
#
#   Python files:
#     approxhaynetstuff_nonparallel.py:
#       Tools for setting the synaptic conductances in a (non-parallelized) reduced-morphology neuron
#
#   Data files:
#     pars_withmids_combfs_final.sav
#       The file containing values of parameters obtained from the optimizations during steps 1-4
# 
# Usage (example):
#   import simseedburst_func_nonparallel_nonadaptive_allions
#   from pylab import *
#   data = simseedburst_func_nonparallel_nonadaptive_allions.simseedburst_func()
#   times = data[0]; vSoma = data[1]; spikes = data[2]
#   f,axarr = subplots(1,1)
#   axarr.plot(times, vSoma[0])
#   axarr.plot(spikes[0], spikes[1],'r.')
#   f.savefig("Vrec.eps")
#
# 29.11.2016: Modified to support the group synapses. Removed support for specialized scaling of background synapses. Added argument gNoiseCoeff for linear background synaptic scaling.


from neuron import h
import numpy
import time
from pylab import *
import scipy.io
import pickle
import sys
import approxhaynetstuff_nonparallel
import resource

# simseedburst_func
#
# The main function for simulating the network (or single-cell) model
#
# Input:
#   Nmc: Number of cells
#   tstop: Length of simulation (biological time, in milliseconds)
#   rdSeed: Random number seed
#   Econ: Synaptic excitatory conductances, in uS (will be multiplied by the synaptic weights)
#   Icon: Synaptic inhibitory conductances, in uS (will be multiplied by the synaptic weights)
#   nseg: Number of subsegments per compartment
#   rateCoeff: The factor for background synaptic rates. By default, f_exc = 0.72 Hz, f_inh = 7.0 Hz
#   gNoiseCoeff: The factor for background synaptic weights. The default value is XXX which produced the same spiking frequency in a single cell as the full single-cell model (Hay & Segev 2015)
#   gSynCoeff: The factor for intra-network weights. The default value is 2.0 which produced the same spiking frequency in the network as the full model (Hay & Segev 2015)
#   Ncells2save: Number of cells for which the membrane potential is saved in high resolution
#   sparsedt: Resolution for the sparsely measured variables (ionic conductances etc.)
#   Nsyns2save: Number of synapses for which the synaptic current is saved
#
# Output:
#   data = [times,vSoma,spikes,sparseTimes,caSoma,skSoma,cahvaSoma,calvaSoma,natSoma,napSoma,ihSoma,kv31Soma,ktSoma,kpSoma,vDend,caDend,skDend,cahvaDend,calvaDend,natDend,ihDend,kv31Dend,Is,cellsSynRecorded,synIndsRecorded]
#     times: Vector of high-resolution time instances (ms)
#     vSoma: Somatic membrane potentials (high-res) (mV)
#     spikes: Spike times in form [spike_t_vec, cell_vec], where spike_t_vec is an array of spike times (ms) and cell_vec is and array of the neuron IDs that fired
#     sparseTimes: Vector of low-resolution time instances (ms)
#     caSoma: Somatic [Ca2+]_i (mM)
#     skSoma,cahvaSoma,calvaSoma,natSoma,napSoma,ihSoma,kv31Soma,ktSoma,kpSoma: Somatic ion-channel currents (mA/cm2)
#     vDend: Apical membrane potential (mV)
#     caDend: Apical [Ca2+]_i (mM)
#     skDend,cahvaDend,calvaDend,natDend,ihDend,kv31Dend: Apical ion-channel currents (mA/cm2)
#     Is: Synaptic currents (nA)
#     cellsSynRecorded: cell IDs telling which cells the recorded synapses were from
#     synIndsRecorded: synapse IDs of the recorded synapses
def simseedburst_func(Nmc=1,tstop=10200,rdSeed=1,Econ=0.0004,Icon=0.001,nseg=5,rateCoeff=1.0,gNoiseCoeff=1.0,gSynCoeff=2.0,Ncells2save=1,sparsedt=1.0,Nsyns2save=1,connM=[],gToBlock=[],blockEfficiency=0.0):
  myrandsavemat = int(100000*gSynCoeff+1000*Nmc+rdSeed)
  coeffCoeffs = [[0.25,0],[0.125,0],[0.5,0],[0.5,1.0/3],[0.5,2.0/3],[0.5,1.0],[-0.25,0],[-0.125,0],[-0.5,0]]

  filename = 'pars_withmids_combfs_final'
  unpicklefile = open(filename+".sav", 'r')
  unpickledlist = pickle.load(unpicklefile)
  unpicklefile.close()
  print filename+'.sav loaded successfully'
  par_names = unpickledlist[0]
  par_values = unpickledlist[1]
  paramdict = {}
  for i in range(0,len(par_names)):
    paramdict[par_names[i]] = par_values[i]

  #Maybe this will decrease memory leak:
  #for s in h.allsec():
  #  h.delete_section() #(no it didn't... still loses about 16MB per simulation for a single modeled cell)
  h("""
{load_file("stdlib.hoc")}
{load_file("stdrun.hoc")}

initialization_tstart = startsw()

strdef fileName
objref fileObj

fileObj = new File()

rdSeed = """+str(rdSeed)+"""
Nmc = """+str(Nmc)+"""
connectivity = 1

tstop = """+str(tstop)+"""
v_init = -75
rcpWeightFactor = 1.5 // the factor by which reciprocal weights are stronger than unidirectional weights
pT2Tr = 0.06 //probability of reciprocating an existing connection to another L5bPC
pT2T = 0.13 //probability of a L5bPC being connected to another L5bPC
Econ = """+str(Econ)+""" //excitatory synaptic conductance
Icon = """+str(Icon)+""" //inhibitory synaptic conductance
gNoiseCoeff = """+str(gNoiseCoeff)+""" // scaling of background synaptic conductances
NcontE = 5 // number of excitatory synaptic contacts per connection
NsynE = 10000 // number of excitatory synapses
NsynI = 2500  // number of inhibitory synapses
rateE = """+str(0.72*rateCoeff)+""" // average rate of presynaptic excitatory cells
rateI = """+str(7.0*rateCoeff)+""" // average rate of presynaptic inhibitory cells
mainBifurcation = 650

Ncells2save = """+str(Ncells2save)+"""
sparsedt = """+str(sparsedt)+""" // recordings of [Ca], I_SK and vApical are done with low temporal resolution to save memory
gSynCoeff = """+str(gSynCoeff)+"""
""")
  print "Params OK!"

  h("""
{load_file("models/TTC.hoc")}

objref MC_TTC
objref sl //synaptic locations list

objref rds1,rds2,rds3
{rds1 = new Random(1000*rdSeed)}
{rds1.uniform(0,1)} 

objref conMat
conMat = new Matrix(Nmc,Nmc)

for(i=0;i<Nmc;i+=1){
        conMat.x[i][i]=0
}
""")
  if len(connM) == 0:
    h("""
for(i=0;i<(Nmc-2);i+=1){
        for(j=(i+1);j<Nmc;j+=1){
                if (connectivity){
                        pcon = rds1.repick()
                        if (pcon<pT2Tr){
                                conMat.x[i][j]=rcpWeightFactor*gSynCoeff
                                conMat.x[j][i]=rcpWeightFactor*gSynCoeff
                        } else {
                                if (pcon<(pT2Tr + 0.5*pT2T)){
                                        conMat.x[i][j]=gSynCoeff
                                        conMat.x[j][i]=0
                                } else {
                                        if (pcon<(pT2Tr + pT2T)){
                                                conMat.x[i][j]=0
                                                conMat.x[j][i]=gSynCoeff
                                        } else {
                                                conMat.x[i][j]=0
                                                conMat.x[j][i]=0
                                        }
                                }
                        }
                } else {
                        conMat.x[i][j]=0
                        conMat.x[j][i]=0
                }
        }
}
""")
  else:
    for i in range(0,Nmc):
      for j in range(0,Nmc):
        if connM[i][j]:
          h("conMat.x["+str(i)+"]["+str(j)+"]="+str(gSynCoeff*connM[i][j])) # Remember that conMat.x[i][j] is connection FROM j TO i
  print "Connectivity OK!"

  h("""
//==================== presynaptic spike trains ====================

objref preTrainList, cells
cells = new List() 
""")

  h("""
strdef treename
objref NsynsE, NsynsI
""")

  for i in range(0,Nmc):
    h("""
  i = """+str(i)+"""
  {cells.append(new TTC())}
  {cells.o[i].initRand(1000*rdSeed+i)}
  {cells.o[i].setnetworkparameters(rcpWeightFactor,Econ,Icon,NsynE,NsynI,NcontE,1.0,1.0,1.0,gNoiseCoeff)}
""")
  approxhaynetstuff_nonparallel.setparams(paramdict,Nmc)

  h("""
  lengthA = cells.o[0].apic[0].L + cells.o[0].apic[1].L
  lengthB = cells.o[0].dend.L
  pA = lengthA/(lengthA + lengthB)
""")

  for i in range(0,Nmc):
    h("""
  i = """+str(i)+"""
  {NsynsE = new List()}
  {NsynsI = new List()}
  for i1 = 0, 2 {
    {NsynsE.append(new Vector("""+str(nseg)+"""))}
    {NsynsI.append(new Vector("""+str(nseg)+"""))}
  }

  for(i1=0;i1<(NsynE+NsynI);i1+=1){
    if (cells.o[i].rd1.repick()<pA){
      treename = "apic"
      compInd = 1
    } else {
      treename = "dend"
      compInd = 0
    }
    sl = cells.o[i].locateSites(treename,cells.o[i].rd1.repick()*cells.o[i].getLongestBranch(treename))
    sitenum = int((sl.count()-1)*cells.o[i].rd1.repick())
    compInd = compInd + sl.o[sitenum].x[0] // if we are at apical, and sl.o[sitenum].x[0]=1, then compInd = 2, otherwise 1 at apical, and 0 at basal
    segInd = int(sl.o[sitenum].x[1]*"""+str(nseg)+""")
    if (i1<NsynE) {
      NsynsE.o[compInd].x[segInd] = NsynsE.o[compInd].x[segInd] + 1
    } else {
      NsynsI.o[compInd].x[segInd] = NsynsI.o[compInd].x[segInd] + 1
    }
  }
  {cells.o[i].distributeSyn(NsynsE,NsynsI)}

  {preTrainList = new List()}
  {rds2 = new Random(1000*rdSeed+i)}//random for presynaptic trains
  {rds3 = new Random(1000*rdSeed+i)}//random for presynaptic trains
  {rds2.negexp(1/rateE)}
  {rds3.negexp(1/rateI)}

  for(compInd=0;compInd<3;compInd+=1){
    for(segInd=0;segInd<"""+str(nseg)+""";segInd+=1){
      {preTrainList.append(new Vector())}
      pst=0 //presynaptic spike time
      if(NsynsE.o[compInd].x[segInd]==0) {
        print "Warning: NsynsE.o[",compInd,"].x[",segInd,"] = 0!!!!"
        pst = 1e6
      }
      while(pst < tstop){
        pst+= 1000*rds2.repick()/NsynsE.o[compInd].x[segInd]
        {preTrainList.o[preTrainList.count()-1].append(pst)}
      }
    }
  }
  for(compInd=0;compInd<3;compInd+=1){
    for(segInd=0;segInd<"""+str(nseg)+""";segInd+=1){
      {preTrainList.append(new Vector())}
      pst=0 //presynaptic spike time
      if(NsynsI.o[compInd].x[segInd]==0) {
        print "Warning: NsynsI.o[",compInd,"].x[",segInd,"] = 0!!!!"
        pst = 1e6
      }
      while(pst < tstop){
        pst+= 1000*rds3.repick()/NsynsI.o[compInd].x[segInd]
        {preTrainList.o[preTrainList.count()-1].append(pst)}
      }
    }
  }
  {cells.o[i].setpretrains(preTrainList)}
  {cells.o[i].queuePreTrains()}
""")

  print "Spike trains OK!"

  h("""
thisCa = 0.0001
for(i=0;i<Nmc;i+=1){
  thisCa = cells.o[i].soma.minCai_CaDynamics_E2
}
""")
  thisCa = h.thisCa


  myMechs = ['Ca_HVA','Ca_LVAst','Ih','Im','K_Pst','K_Tst','NaTa_t','Nap_Et2','SK_E2','SKv3_1','']
  for iblock in range(0,len(gToBlock)):
    for iMech in range(0,len(myMechs)):
      if gToBlock[iblock] in myMechs[iMech]:
        break
    if iMech <= 9:
      print("forall if(ismembrane(\""+str(myMechs[iMech])+"\")) { g"+str(myMechs[iMech])+"bar_"+str(myMechs[iMech])+" = g"+str(myMechs[iMech])+"bar_"+str(myMechs[iMech])+" * "+str(blockEfficiency)+" }")
      h("forall if(ismembrane(\""+str(myMechs[iMech])+"\")) { g"+str(myMechs[iMech])+"bar_"+str(myMechs[iMech])+" = g"+str(myMechs[iMech])+"bar_"+str(myMechs[iMech])+" * "+str(blockEfficiency)+" }")
    else:
      print "Error: No mechanism recognized"

  h("""
v_init = -80
cai0_ca_ion = thisCa
objref syninds, conMatRows

for(i=0;i<Nmc;i+=1){
  cells.o[i].insertMCcons(conMat.getcol(i))
}

{syninds = new Vector()}
{conMatRows = new List()}
objref netcon, mynetconlist
{mynetconlist = new List()}

for(i=0;i<Nmc;i+=1){
        syninds.append(2*3*"""+str(nseg)+""") // 3 compartments with (group-)synapses, each with "nseg" segments, and both E and I synapses included -> 2*3*nseg
}


// appending the microcircuit connections
for(i=0;i<Nmc;i+=1){
        conMatRows.append(new Vector())
        for(j=0;j<Nmc;j+=1){
                conMatRows.o[i].insrt(j,conMat.x[j][i])
                if (conMat.x[j][i] != 0){
                        for(jj=0;jj<NcontE;jj+=1){
                                syninds.x[i] +=1
                                print "c[", j,"].s c[", i,"].s[", syninds.x[i]-1, "]"
                                cells.o[j].soma netcon = new NetCon(&v(0.5),cells.o[i].synlist.o[syninds.x[i]-1],-20,0.5,1.0)
                                mynetconlist.append(netcon)
                        }
                }
        }
}
""")
  h("forall nseg="+str(nseg))
  print "Syninds OK!"

  h("""
objref st1

objref stList
stList = new List()

""")
  print "stList OK!"

  h("""
objref vSomaList, tvecList, caSomaList, skSomaList, cahvaSomaList, calvaSomaList
objref natSomaList, napSomaList, ihSomaList, kv31SomaList, ktSomaList, kpSomaList, IList
objref apcvecList, apcList, nil, spikes, spikedCells
{spikes = new Vector()}
{spikedCells = new Vector()}


{apcvecList = new List()}
{apcList = new List()}
{vSomaList = new List()}
{caSomaList = new List()}
{skSomaList = new List()}
{cahvaSomaList = new List()}
{calvaSomaList = new List()}
{natSomaList = new List()}
{napSomaList = new List()}
{ihSomaList = new List()}
{kv31SomaList = new List()}
{ktSomaList = new List()}
{kpSomaList = new List()}
{IList = new List()}
{tvecList = new List()}
""")

  Nsyns = numpy.array(h.syninds)-2*3*nseg
  cumpNsyns = numpy.cumsum(Nsyns)/numpy.sum(Nsyns)
  randVec = [rand() for x in range(0,Nsyns2save)]
  if sum(Nsyns) > 0:
    cellsSynRecorded = [next(i for i,x in enumerate(cumpNsyns) if x > randVec[j]) for j in range(0,Nsyns2save)]
    synIndsRecorded = [int(2*3*nseg+rand()*Nsyns[i]) for i in cellsSynRecorded]
  else:
    cellsSynRecorded = []
    synIndsRecorded = []

  for i in range(0,Nmc):
    h("""
  i = """+str(i)+"""
  {caSomaList.append(new Vector())}
  {skSomaList.append(new Vector())}
  {cahvaSomaList.append(new Vector())}
  {calvaSomaList.append(new Vector())}
  {natSomaList.append(new Vector())}
  {napSomaList.append(new Vector())}
  {ihSomaList.append(new Vector())}
  {kv31SomaList.append(new Vector())}
  {ktSomaList.append(new Vector())}
  {kpSomaList.append(new Vector())}
  {tvecList.append(new Vector())}
  {sl = cells.o[i].locateSites(\"apic\",mainBifurcation)}
  maxdiam = 0
  for(i1=0;i1<sl.count();i1+=1){
          dd1 = sl.o[i1].x[1]
          dd = cells.o[i].apic[sl.o[i1].x[0]].diam(dd1)
          if (dd > maxdiam) {
                  j = i1
                  maxdiam = dd
          }
  }
  access cells.o[i].soma
  {caSomaList.o[caSomaList.count()-1].record(&cai(0.5),sparsedt)}
  {skSomaList.o[skSomaList.count()-1].record(&ik_SK_E2(0.5),sparsedt)}
  {cahvaSomaList.o[skSomaList.count()-1].record(&ica_Ca_HVA(0.5),sparsedt)}
  {calvaSomaList.o[skSomaList.count()-1].record(&ica_Ca_LVAst(0.5),sparsedt)}
  {natSomaList.o[skSomaList.count()-1].record(&ina_NaTa_t(0.5),sparsedt)}
  {napSomaList.o[skSomaList.count()-1].record(&ina_Nap_Et2(0.5),sparsedt)}
  {ihSomaList.o[skSomaList.count()-1].record(&ihcn_Ih(0.5),sparsedt)}
  {kv31SomaList.o[skSomaList.count()-1].record(&ik_SKv3_1(0.5),sparsedt)}
  {ktSomaList.o[skSomaList.count()-1].record(&ik_K_Tst(0.5),sparsedt)}
  {kpSomaList.o[skSomaList.count()-1].record(&ik_K_Pst(0.5),sparsedt)}
""")
    indSynIndsRecorded = [ix for ix,x in enumerate(cellsSynRecorded) if x==i]
    for isyn in range(0,len(indSynIndsRecorded)):
      h("""
  {IList.append(new Vector())}
  {IList.o[IList.count()-1].record(&cells.o[i].synlist.o["""+str(synIndsRecorded[indSynIndsRecorded[isyn]])+"""].i, sparsedt)}
  """)

    if i < Ncells2save:
      h("""
  {vSomaList.append(new Vector())}
  {vSomaList.o[vSomaList.count()-1].record(&v(0.5),dt)}
""")


  for i in range(0,Nmc):
    h("""
  i = """+str(i)+"""
  access cells.o[i].soma
  {apcList.append(new APCount(0.5))}
  {apcvecList.append(new Vector())}
  apcList.o[apcList.count()-1].thresh= -40
  {apcList.o[apcList.count()-1].record(apcvecList.o[apcList.count()-1])}
  {netcon = new NetCon(&v(0.5), nil)}
  netcon.threshold = -20
  {netcon.record(spikes, spikedCells)  }
""")

  print "Connection matrix:"
  print str(numpy.array(h.conMatRows))

  h("""
stdinit()
sim_tstart = startsw()
initializationtime = (sim_tstart-initialization_tstart)/3600
print \"Initialization completed. Initialization took \", initializationtime, \" hours\\n\"
""")

  h("""
print \"Starting simulation\\n\"
run()

simruntime = (startsw() - sim_tstart)/3600
print \"Simulation took \", simruntime, \" hours\\n\"
""")
  print "Simulation OK!"
  print h.tstop

  vSoma = numpy.array(h.vSomaList)
  times = [h.dt*x for x in range(0,len(vSoma[0]))]
  spikes = [numpy.array(h.spikes), [x-min(numpy.array(h.spikedCells)) for x in numpy.array(h.spikedCells)]]
  caSoma = numpy.array(h.caSomaList)
  skSoma = numpy.array(h.skSomaList)
  cahvaSoma = numpy.array(h.cahvaSomaList)
  calvaSoma = numpy.array(h.calvaSomaList)
  natSoma = numpy.array(h.natSomaList)
  napSoma = numpy.array(h.napSomaList)
  ihSoma = numpy.array(h.ihSomaList)
  kv31Soma = numpy.array(h.kv31SomaList)
  ktSoma = numpy.array(h.ktSomaList)
  kpSoma = numpy.array(h.kpSomaList)
  Is = numpy.array(h.IList)
  sparseTimes = [sparsedt*x for x in range(0,len(caSoma[0]))]

  #Maybe this will decrease memory leak:
  #for listName in ["vSomaList", "caSomaList", "skSomaList","cahvaSomaList","calvaSomaList","natSomaList","napSomaList","ihSomaList","kv31SomaList","ktSomaList","kpSomaList","vApicalList","caApicalList","skApicalList","cahvaApicalList","calvaApicalList","natApicalList","ihApicalList","kv31ApicalList","IList"]:
  #  h(listName+".remove_all()")

  print "runworkers OK!"

  return [times,vSoma,spikes,sparseTimes,caSoma,skSoma,cahvaSoma,calvaSoma,natSoma,napSoma,ihSoma,kv31Soma,ktSoma,kpSoma,Is,cellsSynRecorded,synIndsRecorded]


