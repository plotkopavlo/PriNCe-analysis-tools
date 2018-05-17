from os import path
import numpy as np


def setup_run():
    import cPickle as pickle
    lustre = path.expanduser("~/lustre/")
    with open(lustre + 'prince_run_PSB.ppo', 'rb') as thefile:
        prince_run = pickle.load(thefile)
    return prince_run


def single_run(setup, gamma=None, rmax=None, rscale=None):
    prince_run = setup

    from analyzer.optimizer import UHECRWalker
    from analyzer.spectra import auger2015, Xmax2015
    walker = UHECRWalker(prince_run, auger2015, Xmax2015)

    species = [
        101, 402, 703, 904, 1105, 1206, 1407, 1608, 1909, 2010, 2311, 2412,
        2713, 2814, 3115, 3216, 3517, 4018, 3919, 4020, 4521, 4822, 5123, 5224,
        5525, 5626
    ]
    print [s for s in species if s not in prince_run.spec_man.known_species]
    res = walker.compute_gridpoint(species, **{
        'rmax': rmax,
        'gamma': gamma,
        'm': 'flat',
        'sclass': 'auger',
        'initial_z': 1.
    })
    return res


lustre = path.expanduser('~/lustre')
base = path.abspath(__file__)
config = {
    # Base folder informations
    'project_tag':
    'scan2D_all_elems_PSB',
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
    400,
    'hours per job':
    5,
    'max memory GB':
    3,
    'paramlist': (
        ('gamma', np.linspace(-1.5, 2.5, 81)),
        ('rmax', np.logspace(8.5, 11.5, 61)),
        # ('rscale', np.linspace(-1, 2, 61)),
    ),
}

if __name__ == '__main__':
    # Parse the run arguments
    from analyzer.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()
