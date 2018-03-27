import os
import sys
sys.path.append('../PriNCe/')
# sys.path.append('../pr_analyzer/')
import cPickle as pickle

lustre = os.path.expanduser("~/lustre/")
with open(lustre + 'prince_run_PSB.ppo','rb') as thefile:
    prince_run = pickle.load(thefile)

from analyzer.optimizer import UHECRWalker

from analyzer.spectra import auger2015, Xmax2015, TA2015
walker = UHECRWalker(prince_run, auger2015, Xmax2015)

rmax = 10**12.
gamma = 2.5
pids = [101,402,1407,2814,5626]
source_params = (rmax, gamma)

chain = walker.run_mcmc(source_params, pids, mpi=True)

lustre = os.path.expanduser("~/lustre/")
with open(lustre + 'sampled_chain_proton.ppo','wb') as thefile:
    pickle.dump(chain, thefile, protocol = -1)

print 'finished sampling'