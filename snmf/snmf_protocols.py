#Protocols for fitting the non-synaptic conductances in four steps.
#Defines which quantities are fitted at each step, what stimuli are for each objective function, etc.
#
#Tuomo Maki-Marttunen, 2015-2016
#

from neuron import h

#Names and ranges of each varied model parameter. Grouped into four fitting steps.
snmf_variables =   [ [ ["L_soma", 11.58, 46.34],               #First step parameters: L_soma
                       ["L_dend", 141.06, 564.26],             #L_basal   
                       ["L_apic[0]", 325.0, 1300.0],           #L_apical  
                       ["L_apic[1]", 325.26, 1301.06],         #L_apical2 
                       ["Ra_soma", 20, 500],                   #Ra_soma   
                       ["Ra_dend", 10, 1000],                   #Ra_basal  
                       ["Ra_apic[0]", 10, 1000],                #Ra_apical 
                       ["Ra_apic[1]", 10, 1000],                #Ra_apical2
                       ["cm_soma", 0.5, 2.0],                  #Cm_soma   
                       ["cm_dend", 0.5, 4.0],                  #Cm_basal  
                       ["cm_apic[0]", 0.5, 4.0],               #Cm_apical 
                       ["cm_apic[1]", 0.5, 4.0],               #Cm_apical2
                       ["g_pas_soma", 0.00002, 0.0001],       #gpas_soma
                       ["g_pas_dend", 0.000015, 0.0001],      #gpas_basal
                       ["g_pas_apic[0]", 0.000015, 0.0001],   #gpas_apical
                       ["g_pas_apic[1]", 0.000015, 0.0001] ], #gpas_apical2
                     [ ["ehcn_Ih_*", -55, -35],                #Second step parameters: E_h (Not fitted in Table 2. Used +-10 mV)
                       ["gIhbar_Ih_soma", 0, 0.0008],          #Ih_soma (Not fitted in Table 2. Used 4-fold wrt Hay value)       
                       ["gIhbar_Ih_dend", 0, 0.0008],          #Ih_basal (Not fitted in Table 2. Used 4-fold wrt Hay value)      
                       ["gIhbar_Ih_apic[0]", 0, 0.008],        #Ih_apical (Not fitted in Table 2. Used 8-fold wrt mean Hay value)
                       ["gIhbar_Ih_apic[1]", 0, 0.016],        #Ih_apical2 (Not fitted in Table 2. Used 16-fold wrt mean Hay value)
                       ["g_pas_soma", 0.00002, 0.0001],       #gpas_soma
                       ["g_pas_dend", 0.000015, 0.0001],      #gpas_basal
                       ["g_pas_apic[0]", 0.000015, 0.0001],   #gpas_apical
                       ["g_pas_apic[1]", 0.000015, 0.0001] ], #gpas_apical2
                     [ ["gCa_HVAbar_Ca_HVA_soma", 0, 0.001],           #CaHVA_soma
                       ["gCa_LVAstbar_Ca_LVAst_soma", 0, 0.01],        #CaLVA_soma
                       ["gamma_CaDynamics_E2_soma", 0.0005, 0.05],     #gamma_soma
                       ["decay_CaDynamics_E2_soma", 20.0, 1000.0],     #decay_soma
                       ["gSK_E2bar_SK_E2_soma", 0, 0.1],               #SK_soma   
                       ["gCa_HVAbar_Ca_HVA_apic[0]", 0, 0.0025],       #CaHVA_apical
                       ["gCa_LVAstbar_Ca_LVAst_apic[0]", 0, 0.1],      #CaLVA_apical
                       ["gamma_CaDynamics_E2_apic[0]", 0.0005, 0.05],  #gamma_apical
                       ["decay_CaDynamics_E2_apic[0]", 20.0, 200.0],   #decay_apical
                       ["gSK_E2bar_SK_E2_apic[0]", 0, 0.005],          #SK_apical   
                       ["gCa_HVAbar_Ca_HVA_apic[1]", 0, 10*0.0025],    #CaHVA_apical2 (Use 10-fold maximal value to take into account the possible hot region)
                       ["gCa_LVAstbar_Ca_LVAst_apic[1]", 0, 10*0.1],   #CaLVA_apical2 (Use 10-fold maximal value to take into account the possible hot region)
                       ["gamma_CaDynamics_E2_apic[1]", 0.0005, 0.05],  #gamma_apical2
                       ["decay_CaDynamics_E2_apic[1]", 20.0, 200.0],   #decay_apical2
                       ["gSK_E2bar_SK_E2_apic[1]", 0, 0.005] ],        #SK_apical2   
                     [ ["gNaTa_tbar_NaTa_t_soma", 0, 4.0],           #Nat_soma
                       ["gNap_Et2bar_Nap_Et2_soma", 0, 0.01],        #Nap_soma  
                       ["gK_Tstbar_K_Tst_soma", 0, 0.1],             #Kt_soma   
                       ["gK_Pstbar_K_Pst_soma", 0, 1.0],             #Kp_soma   
                       ["gSKv3_1bar_SKv3_1_soma", 0, 2.0],           #Kv31_soma 
                       ["gImbar_Im_apic[0]", 0, 0.0005],                #Im_apical 
                       ["gNaTa_tbar_NaTa_t_apic[0]", 0, 0.02],          #Nat_apical
                       ["gSKv3_1bar_SKv3_1_apic[0]", 0, 0.02],          #Kv31_apical
                       ["gImbar_Im_apic[1]", 0, 0.0005],                #Im_apical2   
                       ["gNaTa_tbar_NaTa_t_apic[1]", 0, 0.02],            #Nat_apical2  
                       ["gSKv3_1bar_SKv3_1_apic[1]", 0, 0.02] ] ]       #Kv31_apical2 


# Stimulus mediators (names and types of point processes):
# 0: Use the IClamp named "st1" at soma(0.5)
# 1: Use the EPSP-like stimulus named "syn1" at a segment on apical dendrite 620 um from the soma
#                   NAME    TYPE      WHERE    AT A FIXED POS?   AMPLITUDE NAME
#                                              OR AT A DISTANCE                
#                                              FROM SOMA?                      
stimulus_types = [ ["st1",  "IClamp", "soma", ["fixed", 0.5],    "amp" ],
                   ["syn1", "epsp",   "apic", ["distance", 620], "imax" ] ]

# Types of data storage (what data to pass on to the objective functions):
# 0: The membrane potential (or Ca concentration) at t=13000 ms
# 1: The maximum of membrane potential on interval 10000..10200 ms
# 2: The membrane potential time series from 50 ms prior to beginning of stimulus to 200 ms post, use a 5-ms resolution
# 3: The membrane potential time series from 50 ms prior to beginning of stimulus to 200 ms post, use a 1-ms resolution
# 4: The membrane potential time series from 50 ms prior to beginning of stimulus to 200 ms post, use a 1-ms resolution, and the exact spike times
# 5: Just the number of spikes during interval 12000...15000 ms
#                   WHAT TYPE OF OUTPUT? VOLTAGE/[CA] AT A
#                   FIXED TIME (LAST TVEC BEFORE GIVEN TIME)
#                   OR MAXIMUM/NSPIKES DURING A GIVEN INTERVAL?
data_storage_types = [ ["fixed", 13000],
                       ["max", [10000,10200] ],
                       ["trace", [9950+5*x for x in range(0,51)] ],
                       ["highrestrace", [9950+x for x in range(0,251)] ],
                       ["highrestraceandspikes", [9950+x for x in range(0,251)] ],
                       ["nspikes", [12000, 15000] ] ]


# Types of stimuli
# 0: Use IClamp ("st1") with onset at 10000 ms and 3000-ms duration 
# 1: Use IClamp ("st1") with onset at 10000 ms and 100-ms duration 
# 2: Use EPSP-like stimulus ("syn1") with onset at 10000 ms, rise time of 0.5 ms and decay time of 5.0 ms
# 3: Use IClamp ("st1") with onset at 10000 ms and 5000-ms duration 
# 4: Use IClamp ("st1") with onset at 10000 ms and 5-ms duration 
#             STIMULUS MEDIATOR
#                LIST OF POINT PROCESS PARAMETERS
stimuli = [ [ 0, [ ["del", 10000], ["dur", 3000] ] ],
            [ 0, [ ["del", 10000], ["dur", 100] ] ],
            [ 1, [ ["onset", 10000], ["tau0", 0.5], ["tau1", 5.0] ] ],
            [ 0, [ ["del", 10000], ["dur", 5000] ] ],
            [ 0, [ ["del", 10000], ["dur", 5] ] ] ]

# Types of recordings
# 0: Membrane potential at soma
# 1: Membrane potential at dendrites
# 2: Calcium concentration at apical dendrite
#                NAME      WHAT   WHERE
#                                   WHICH    AT WHICH
#                                   BRANCH   LOCATIONS
recordings = [ [ "vsoma",  "v",   [ ["soma", [0.5] ] ] ],
               [ "vdend",  "v",   [ ["apic", [0.05*x for x in range(0,21)] ], 
                                    ["dend", [0.05*x for x in range(0,21)] ] ] ],
               [ "cadend", "cai", [ ["apic", [0.05*x for x in range(0,21)] ] ] ] ]

# The setup for the objective functions. The structure consists of four steps, each with 2-4 objective functions, each of which contains
# the list of stimulus indices, the list of amplitudes, the recording index (or list of them if several), and the data storage index.
#                  STIMULUS INDEX
#                          AMPLITUDES
#                                                   RECORDING INDEX
#                                                           DATA STORAGE INDEX
snmf_setup = [ [ [ [0],   [0.5],                    1,      0 ],            #STEP 1: MEMBRANE POTENTIAL STEADY STATE RESPONSE TO A NONZERO LONG DC,
                 [ [2],   [0.5],                    1,      1 ],            #        MEMBRANE POTENTIAL MAXIMUM AS A RESPONSE TO AN EPSP,
                 [ [1],   [-1.0, 0, 1.0],           0,      2 ] ],          #        AND MEMBRANE POTENTIAL TRACE RESPONSE TO A SHORT (100 ms) DC
               [ [ [0],   [0, 0.5, 1.0],            1,      0 ],            #STEP 2: MEMBRANE POTENTIAL STEADY STATE RESPONSE TO A ZERO AND NONZERO LONG DC
                 [ [1],   [0.5, 1.0],               0,      2 ] ],          #        AND MEMBRANE POTENTIAL TRACE RESPONSE TO A SHORT (100 ms) DC
               [ [ [1],   [1.0, 2.0],               0,      3 ],            #STEP 3: MEMBRANE POTENTIAL TRACE RESPONSE TO A SHORT (100 ms) STRONG DC,
                 [ [0],   [0.5, 1.0],               [1, 2], 0 ],            #        MEMBRANE POTENTIAL STEADY STATE RESPONSE TO A LONG DC,
                                                                            #        [CA] STEADY STATE RESPONSE TO A LONG DC,
                 [ [2],   [1.0, 2.0],               [1, 2], 1 ] ],          #        MEMBRANE POTENTIAL MAXIMUM AS A RESPONSE TO AN EPSP,
                                                                            #        AND [CA] MAXIMUM AS A RESPONSE TO AN EPSP
               [ [ [1],   [0.25, 0.5],              0,      4 ],            #STEP 4: MEMBRANE POTENTIAL TRACE RESPONSE TO A 100 ms DC,
                 [ [4],   [1.9],                    0,      4 ],            #        MEMBRANE POTENTIAL TRACE RESPONSE TO A SHORT SOMATIC STIMULUS
                 [ [4,2], [[1.9, 0.5]],             0,      4 ],            #        MEMBRANE POTENTIAL TRACE RESPONSE TO COMBINATION OF SHORT SOMATIC AND APICAL STIMULI
                 [ [3],   [0.78, 1.0, 1.9],         0,      5 ] ] ]         #        NUMBERS OF SPIKES AS A RESPONSE TO A LONG DC


# Data structure telling whether we can stop the evaluation of the objective function if there are no spikes when there should be none
stop_if_no_spikes = [ [ [0], [0], [0, 0, 0] ],                 # Propose that the simulation be stopped and high error values be given to the rest of the
                      [ [0, 0, 0], [0, 0] ],                   # objectives in case there are no spikes when there should be (1's denote such stimuli, 0's
                      [ [0, 0], [0, 0], [0, 0] ],                      # denote stimuli where this is not considered)
                      [ [0, 1], [0], [1], [1, 1, 1] ] ]

# Data structure telling whether we can stop the evaluation of the objective function if there are more spikes than announced
stop_if_more_spikes_or_as_many_as = [ [ [0], [0], [0, 0, 0] ],            # Propose that the simulation be stopped and high error values be given to the rest of the
                                      [ [0, 0, 0], [0, 0] ],              # objectives in case there are more than x spikes when there should be much fewer (0's
                                      [ [0, 0], [0, 0], [0, 0] ],                 # denote stimuli where this is not considered, x > 0 denotes stimuli where this is done
                                      [ [2, 5], [5], [5], [100, 200, 300] ] ]  # when nSpikes >= x)

# Return the full matrix of the step-wise fitting variables
def get_snmf_variables():
  global snmf_variables
  return snmf_variables

# Return the parameter names that should be fixed at the istep:th step
def get_fixed_params(istep):
  global snmf_variables
  fixed_params = []
  for istep2 in range(0,istep):
    for ivar in range(0,len(snmf_variables[istep2])):
      fixed_params.append(snmf_variables[istep2][ivar][0])
  return fixed_params

# Return the parameter names that should be fitted during the istep:th step
def get_variable_params(istep):
  global snmf_variables
  variable_params = []
  if istep < len(snmf_variables):
    for ivar in range(0,len(snmf_variables[istep])):
      variable_params.append(snmf_variables[istep][ivar][0])
  return variable_params
  
# Return the stimulus array
def get_stimuli():
  global stimuli
  return stimuli

# Return the stimulus types array
def get_stimulus_types():
  global stimulus_types
  return stimulus_types

# Return the data storage array
def get_data_storage_types():
  global data_storage_types
  return data_storage_types

# Return the recordings array
def get_recordings():
  global recordings
  return recordings

# Return the stimulus-recording setup array
def get_setup():
  global snmf_setup
  return snmf_setup

def get_nspike_restrictions():
  global stop_if_no_spikes, stop_if_more_spikes_or_as_many_as
  return [stop_if_no_spikes, stop_if_more_spikes_or_as_many_as]




