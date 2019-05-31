# Python file for drawing results from network (N=150) simulations
# Tuomo Maki-Marttunen, 2015-2016
import mytools
from pylab import *
import pickle

Nmc = 150
gSynCoeffs = [1.1, 1.25, 1.4]

spikers_all_all = []
spts_all_all = []
cumFRs_all_all = []
dtFR = 5.0
T = 10200
cols = ['#0000FF','#FF0000','#B3B300']
dimcols = ['#AAAAFF','#FFAAAA','#E5E5AA']

for igsyn in range(0,3):
  spikers_all = []
  spts_all = []
  cumFRs_all = []
  gSynCoeff = gSynCoeffs[igsyn]
  for myseed in range(1,15):
    unpicklefile = open('spikes_nonadaptive_'+str(Nmc)+'_gsyn'+str(gSynCoeff)+'_seed'+str(myseed)+'.sav', 'r')
    unpickledlist = pickle.load(unpicklefile)
    unpicklefile.close()
    spikes = unpickledlist[:]
    spts_all.append(spikes[0])
    spikers_all.append(spikes[1])
    cumFRs_this = [sum([1 for x in spikes[0] if x <= i*dtFR]) for i in range(0,int(T/dtFR))]
    cumFRs_all.append(cumFRs_this[:])
    print "myseed = "+str(myseed)+" analyzed"
  spts_all_all.append(spts_all[:])
  spikers_all_all.append(spikers_all[:])
  cumFRs_all_all.append(cumFRs_all[:])

close("all")
f,axarr = subplots(2,1)
for igsyn in range(0,3):
  axarr[0].plot(spts_all_all[igsyn][0],[150*igsyn+x for x in spikers_all_all[igsyn][0]],'r.',color=cols[igsyn],markersize=1.0)
  for isamp in range(0,14):
    axarr[1].plot([i*dtFR for i in range(0,int(T/dtFR))], cumFRs_all_all[igsyn][isamp], 'r-', color=dimcols[igsyn])
for igsyn in range(0,3):
  axarr[1].plot([i*dtFR for i in range(0,int(T/dtFR))], [mean([cumFRs_all_all[igsyn][isamp][i] for isamp in range(0,14)]) for i in range(0,len(cumFRs_all_all[igsyn][0]))], 'r-', color=cols[igsyn],linewidth=2)

axarr[0].set_xlim([0,10000])
axarr[1].set_xlim([0,10000])
f.savefig("cumFRs.eps")
