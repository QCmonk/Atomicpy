from qChain import *



###############################################
# Compressive Reconstruction of Neural Signal #
###############################################
taus = [0.01, 0.025, 0.05, 0.075]

while True:
  # get number of measurements to use
  meas = np.clip(int(input("Input the number of measurements to use (a number between 1 and 80): ")),1,80)
  # gen index of time event
  taui = np.clip(int(input("Choose an event time (pick a number between 1 and 4): ")),1,4)
  tau = taus[taui-1]
  # get noise level (noise of 1 -> SNR ~= 1 while 0 -> SNR = infty)
  noise = np.clip(float(input("Choose a noise level (pick a number between 0 and 1): ")), 0,1)/100

  # define user parameters for simulation run
  sim_vars = {"measurements":   int(meas),           # number of measurements
              "tau":            float(tau),           # time of pulse - one of 0.05, 0.1, 0.15, 0.2
              "noise":          float(noise)}          # noise in system

  # simulate the system!
  demonstrate(sim_vars)



