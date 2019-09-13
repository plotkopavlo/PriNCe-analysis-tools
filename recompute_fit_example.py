from os import path
import numpy as np


def setup_fit():
    from os import path
    filepath = path.join(config['targetdir'],config['project_tag'],'collected.hdf5')
    input_spec = config['input_spec']
    paramlist = config['paramlist']

    from analyzer.plotter import ScanPlotter
    scan = ScanPlotter(filepath, input_spec, paramlist)

    scan.print_summary()
    return scan


def single_fit(setup, index):
    scan = setup

    m, opt = scan.recompute_fit(index, minimizer_args={'fix_deltaE':True},Emin=6e9)
    print 'chi2:', opt.get_chi2_spectrum(),opt.get_chi2_Xmax(), opt.get_chi2_VarXmax()
    mindetail = m.parameters, m.args, m.values, m.errors
    return m.fval, mindetail

from run import config
base = path.abspath(__file__)
config.update({
    'fit_tag': 'fixed_E',
    'fit_only': True,
    'inputpath':
    base,
    'setup_func': setup_fit,
    'single_run_func': single_fit,
    'njobs':
    60,
    'hours per job':
    1,
    'max memory GB':
    3,
})


if __name__ == '__main__':
    # Parse the run arguments
    from analyzer.cluster import PropagationProject
    project = PropagationProject(config)
    project.run_from_terminal()
