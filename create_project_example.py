from os import path
import numpy as np


def setup_run():
    import cPickle as pickle
    lustre = path.expanduser("~/lustre/")
    with open(lustre + 'prince_run_talys.ppo', 'rb') as thefile:
        prince_run = pickle.load(thefile)
    return prince_run


def single_run(setup, gamma=None, rmax=None):
    prince_run = setup

    from analyzer.optimizer import UHECRWalker
    from analyzer.spectra import auger2015, Xmax2015
    walker = UHECRWalker(prince_run, auger2015, Xmax2015)

    
    # species = [s for s in prince_run.spec_man.known_species is s >= 100]
    species = [101,402,1407,2814,5626]
    res = walker.compute_gridpoint(species, **{
        'rmax': rmax,
        'gamma': gamma,
        'm': 'flat',
    })
    return res


lustre = path.expanduser('~/lustre')
base = path.abspath(__file__)
config = {
    # Base folder informations
    'project_tag':
    'scan2D_flat_talys',
    'targetdir':
    lustre,
    'inputpath':
    base,
    # functions to compute on each grid point
    'setup_func':
    setup_run,
    'single_run_func':
    single_run,
    # Number of jobs and parameterspace
    'njobs':
    450,
    'hours per job':
    5,
    'max memory GB':
    2,
    'paramlist': (
        ('gamma', np.linspace(-1.5, 2.5, 81)),
        ('rmax', np.logspace(8.5, 11.5, 61)),
        # ('m', np.linspace(-6, 6, 61)),
    ),
}

if __name__ == '__main__':
    # Parse the run arguments
    from analyzer.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()
