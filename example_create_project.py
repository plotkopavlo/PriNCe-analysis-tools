from os import path
import numpy as np


def setup_run():
    import pickle as pickle
    lustre = path.expanduser("~/lustre/")
    with open(lustre + 'prince_run_PSB.ppo', 'rb') as thefile:
        prince_run = pickle.load(thefile)
    return prince_run


def single_run(setup, index):
    prince_run = setup

    from analyzer.optimizer import UHECRWalker
    from analyzer.spectra import auger2015, Xmax2015, XRMS2015
    walker = UHECRWalker(prince_run, auger2015, Xmax2015, XRMS2015)

    gamma = config['paramlist'][0][1][index[0]]
    rmax = config['paramlist'][1][1][index[1]]
    m = config['paramlist'][2][1][index[2]]

    species = config['input_spec']
    res = walker.compute_gridpoint(species, **{
        'rmax': rmax,
        'gamma': gamma,
        'm': ('simple', m),
        'sclass': 'auger',
        'initial_z': 1.
    })

    del walker
    return res


lustre = path.expanduser('~/lustre')
base = path.abspath(__file__)
config = {
    # Base folder informations
    'project_tag':
    'scan3D_simple_PSB_more_elem',
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
    8000,
    'hours per job':
    8,
    'max memory GB':
    3,
    'paramlist': (
        ('gamma', np.linspace(-1.5, 2.5, 81)),
        ('rmax', np.logspace(8.5, 11.5, 61)),
        ('m', np.linspace(-6, 6, 61)),
    ),
    'input_spec': [101, 402, 1206, 1608, 2713, 3216, 5626],
}

if __name__ == '__main__':
    # Parse the run arguments
    from analyzer.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()
