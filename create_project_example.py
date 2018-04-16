from os import path
import numpy as np


def setup_run():
    import cPickle as pickle
    lustre = path.expanduser("~/lustre/")
    with open(lustre + 'prince_run_PSB.ppo', 'rb') as thefile:
        prince_run = pickle.load(thefile)
    return prince_run


def single_run(setup, gamma=None, rmax=None):
    prince_run = setup

    from analyzer.optimizer import UHECRWalker
    from analyzer.spectra import auger2015, Xmax2015
    walker = UHECRWalker(prince_run, auger2015, Xmax2015)

    res = walker.lnprob_mc(
        (rmax, gamma), [101, 402, 1407, 2814, 5626], return_blob=True)
    return res


lustre = path.expanduser('~/lustre/Propagation')
base = path.abspath(__file__)
config = {
    'project_tag': 'singularity_test',
    'targetdir': lustre,
    'inputpath': base,
    'njobs': 6,
    'paramlist': {
        'gamma': np.linspace(-1.5, 2.5, 11),
        'rmax': np.logspace(8.5, 11.5, 13)
    },
    'setup_func': setup_run,
    'single_run_func': single_run,
}

if __name__ == '__main__':
    # Parse the run arguments
    from analyzer.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()