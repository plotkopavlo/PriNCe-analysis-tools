from os import path
import numpy as np


def setup_run():
    """Setup function is executed at the start of each job
       The return value is passed to single_run for each index
    """
    import pickle as pickle
    # NOTE: Set the path to your PriNCe kernels below
    path = path.expanduser("~/---/---/")
    with open(path + 'prince_run_xxx.ppo', 'rb') as thefile:
        prince_run = pickle.load(thefile)
    return prince_run

def single_run(setup, index):
    """Single run function is executed for each index
       Every job will loop over a subset of all indices and call this function
       The list of outputs is then stored in .out
    """
    prince_run = setup

    from prince_analysis_tools.optimizer import UHECRWalker
    #NOTE: You can change the year to 2017 and 2019.
    from prince_analysis_tools.spectra import auger2015, Xmax2015, XRMS2015
    walker = UHECRWalker(prince_run, auger2015, Xmax2015, XRMS2015)

    gamma = config['paramlist'][0][1][index[0]]
    rmax = config['paramlist'][1][1][index[1]]
    m = 0.
    #NOTE: use for 3D scanning:
    #m = config['paramlist'][2][1][index[2]]

    print('running with', gamma, rmax, m)
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

# Set the project target path below
lustre = path.expanduser('~/---/---/')
base = path.abspath(__file__)

# set config for jobs options below
# The project will loop over all index combinations in 'paramlist'
# Each job will receive an equal subset of indices to compute
config = {
    # Base folder informations
    'project_tag':
    'scan2D_talys',
    'targetdir':
    lustre,
    'inputpath':
    base,
    #NOTE: functions to compute on each grid point
    'setup_func': 
    setup_run,
    'single_run_func':
    single_run,
    #NOTE: hadronic interaction models for air showers
    #NOTE: there are three options: 'epos' (standart), 'qgsjet'  and 'sibyll' 
    'xmax_model':
    'sibyll',
    #NOTE: Number of jobs and parameterspace
    'njobs':
    500,
    'hours per job':
    8,
    'max memory GB':
    8,
    'paramlist': (
        ('gamma', np.linspace(-1.5, 2.5, 81)),
        ('rmax', np.logspace(8.5, 11.5, 61)),
        #NOTE: use for 3D scanning:
        #('m', np.linspace(-6, 6, 61)),
    ),
    'input_spec': [101, 402, 1407, 2814, 5626],
}

# run this script as python example_create_project.py -[options]
# PropagationProject.run_from_terminal() for all options
if __name__ == '__main__':
    # Parse the run arguments
    from prince_analysis_tools.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()
