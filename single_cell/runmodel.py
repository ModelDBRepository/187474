from neuron import h
from pylab import *

h("""load_file("runmodel.hoc")""")

filenames = ['run_1.dat', 'run_2.dat', 'run_3.dat', 'run_3a.dat']
titles = ['Somatic pulse', 'Apical pulse', 'BAC firing', 'No BAC firing (somatic only)', 'No BAC firing (apical only)']

f,ax = subplots((len(filenames)+1)/2, 2)
cols = ['#FF0000','#0000FF']
extensions = ['', 'fullhay_']
for ifile in range(0,len(filenames)):
  for iextension in range(0,len(extensions)):
    extension = extensions[iextension]
    ts = []
    vs = []
    vdends = []
    with open(extension+filenames[ifile], 'r') as file:        
      for row in file:
        t,v,vdend = row.split()
        ts.append(float(t))
        vs.append(float(v))
        vdends.append(float(vdend))
    ax[ifile/2, ifile%2].plot(ts,vs,color=cols[iextension],linestyle='-')
    l,=ax[ifile/2, ifile%2].plot(ts,vdends,color=cols[iextension],linestyle='--')
    l.set_dashes([2,2,2,2])
  ax[ifile/2, ifile%2].set_xlim([10000,10500])
  ax[ifile/2, ifile%2].set_ylim([-90,40])
  ax[ifile/2, ifile%2].set_xlabel('t (ms)',fontsize=6)
  ax[ifile/2, ifile%2].set_ylabel('V (mV)',fontsize=6)
  ax[ifile/2, ifile%2].set_title(titles[ifile],fontsize=6)
  for tick in ax[ifile/2, ifile%2].yaxis.get_major_ticks()+ax[ifile/2, ifile%2].xaxis.get_major_ticks():
    tick.label.set_fontsize(6)
  if ifile==1:
    ax[0,1].legend(['Soma, reduced morphology','Apical dend, reduced morphology', 'Soma, full morphology','Apical dend, full morphology'],fontsize=6)
f.savefig("runs.eps")


