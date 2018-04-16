import numpy as np
from xmax import XmaxSimple


class UHECROptimizer(object):
    XmaxModel = XmaxSimple(model=XmaxSimple.EPOSLHC)

    def __init__(self,
                 single_results,
                 spectrum,
                 Xmax,
                 ncoids=None,
                 Emin=6e9,
                 norms=None):
        self.Emin = Emin

        # spectral data
        self.spectrum = spectrum
        self.egrid_spectrum = spectrum['energy']

        # Xmax data
        self.Xmax = Xmax
        self.egrid_xmax = Xmax['energy']

        self.lst_res = np.array(single_results)
        if norms is None:
            norms = np.ones_like(self.lst_res, dtype=np.float)
        if ncoids is None:
            ncoids = range(len(self.lst_res))
        self.ncoids = ncoids

        self._create_interpolators()
        self.compute_combined_result(norms)

    def _create_interpolators(self):
        from scipy.interpolate import interp1d

        egrid_spectrum, _ = self.lst_res[0].get_solution_group('CR')
        egrid_xmax, _, _ = self.lst_res[0].get_lnA('CR')
        arr_spectrum = np.zeros((len(self.lst_res), egrid_spectrum.size))
        arr_mean_lnA = np.zeros((len(self.lst_res), egrid_xmax.size))
        arr_var_lnA = np.zeros((len(self.lst_res), egrid_xmax.size))
        for idx, res in enumerate(self.lst_res):
            _, spectrum = res.get_solution_group('CR')
            _, mean_lnA, var_lnA = res.get_lnA('CR')
            arr_spectrum[idx] = spectrum
            arr_mean_lnA[idx] = mean_lnA
            arr_var_lnA[idx] = var_lnA

        self.intp_spectrum = interp1d(
            egrid_spectrum,
            arr_spectrum,
            axis=1,
            fill_value=(0., 0.),
            bounds_error=False)
        self.intp_mean_lnA = interp1d(
            egrid_xmax,
            arr_mean_lnA,
            axis=1,
            fill_value=(0., 0.),
            bounds_error=False)
        self.intp_var_lnA = interp1d(
            egrid_xmax,
            arr_var_lnA,
            axis=1,
            fill_value=(0., 0.),
            bounds_error=False)

    def _intpolate_scipy(self, deltaE):
        self.arr_spectrum = self.intp_spectrum(self.egrid_spectrum *
                                               (1 - deltaE))
        self.arr_spec_lnA = self.intp_spectrum(self.egrid_xmax * (1 - deltaE))
        self.arr_mean_lnA = self.intp_mean_lnA(self.egrid_xmax * (1 - deltaE))
        self.arr_var_lnA = self.intp_var_lnA(self.egrid_xmax * (1 - deltaE))

    def compute_combined_result(self, norms, deltaE=0.):
        self._intpolate_scipy(deltaE)
        # get the averages from subsets by weighting with the norms
        spectrum = (norms[:, np.newaxis] * self.arr_spectrum).sum(axis=0)
        mean_lnA = (
            norms[:, np.newaxis] * self.arr_spec_lnA * self.arr_mean_lnA
        ).sum(axis=0) / (norms[:, np.newaxis] * self.arr_spec_lnA).sum(axis=0)
        var_lnA = (norms[:, np.newaxis] * self.arr_spec_lnA *
                   (self.arr_var_lnA + self.arr_mean_lnA**2)).sum(axis=0) / (
                       norms[:, np.newaxis] * self.arr_spec_lnA
                   ).sum(axis=0) - mean_lnA**2

        self.res_spectrum = spectrum
        self.res_xmax = self.XmaxModel.get_mean_Xmax(mean_lnA, self.egrid_xmax)
        self.res_var_xmax, _ = np.sqrt(
            self.XmaxModel.get_sigma2_Xmax(mean_lnA, var_lnA, self.egrid_xmax))

    def get_chi2_spectrum(self, norms=None):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms)

        sl = np.where(self.egrid_spectrum > self.Emin)

        delta = self.res_spectrum[sl] - self.spectrum['spectrum'][sl]
        error = np.where(self.res_spectrum[sl] > self.spectrum['spectrum'][sl],
                         self.spectrum['upper_err'][sl],
                         self.spectrum['lower_err'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_Xmax(self, norms=None):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms)

        sl = np.where(self.egrid_xmax > self.Emin)

        delta = self.res_xmax[sl] - self.Xmax['Xmax'][sl]
        error = np.where(self.res_xmax[sl] > self.Xmax['Xmax'][sl],
                         self.Xmax['statXmax'][sl], self.Xmax['statXmax'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_VarXmax(self, norms=None):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms)

        sl = np.where(self.egrid_xmax > self.Emin)

        delta = self.res_var_xmax[sl] - self.Xmax['XRMS'][sl]
        error = np.where(self.res_var_xmax[sl] > self.Xmax['XRMS'][sl],
                         self.Xmax['statXRMS'][sl], self.Xmax['statXRMS'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_total(self, norms=None, deltaE=0.):
        if self.lst_res is not None and norms is not None:
            self.compute_combined_result(norms, deltaE)

        return sum([
            self.get_chi2_spectrum(),
            self.get_chi2_Xmax(),
            self.get_chi2_VarXmax()
        ])

    def fit_data_minuit(self,
                        spectrum_only=False,
                        minimizer_args={},):
        from iminuit import Minuit

        def chi2(deltaE, *norms):
            norms = np.array(norms)
            result = self.get_chi2_total(norms=norms, deltaE=deltaE)
            return result

        init_norm = self.spectrum['spectrum'][14] / self.res_spectrum[
            14] / len(self.ncoids)
        arg_names = ['deltaE'] + ['norm{:}'.format(pid) for pid in self.ncoids]
        start = [0.] + [init_norm] * len(self.ncoids)
        error = [0.1] + [init_norm / 2] * len(self.ncoids)
        limit = [(-0.14, 0.14)] + [(1e20, 1e40)] * len(self.ncoids)

        params = {}
        params.update({name: val for name, val in zip(arg_names, start)})
        params.update(
            {'error_' + name: val
             for name, val in zip(arg_names, error)})
        params.update(
            {'limit_' + name: val
             for name, val in zip(arg_names, limit)})
        params.update(minimizer_args)

        m = Minuit(chi2, forced_parameters=arg_names, **params)
        m.migrad()
        return m

class UHECRWalker(object):
    def __init__(self, prince_run, spectrum, xmax, progressbar=False):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax
        self.progressbar = progressbar

    def compute_models(self, source_params, particle_ids):
        """
        Compute the results corresponding to source_params for each particle id individually and return a list
        """
        from prince.solvers import UHECRPropagationSolver
        from prince.cr_sources import AugerFitSource

        lst_models = []
        rmax, gamma = source_params
        for ncoid in particle_ids:

            solver = UHECRPropagationSolver(
                initial_z=1., final_z=0., prince_run=self.prince_run)

            params = {
                ncoid: (gamma, rmax, 1.),
            }
            source = AugerFitSource(self.prince_run, params=params, norm=1e-80)
            solver.add_source_class(source)
            solver.set_initial_condition()
            solver.solve(
                dz=1e-3,
                verbose=False,
                full_reset=False,
                progressbar=self.progressbar)

            lst_models.append(solver.res)

        # return the results only
        return lst_models

    def extract_minuit(self, minres):
        return minres.parameters, minres.args, minres.values, minres.errors

    def lnprob_mc(self,
                  source_params,
                  particle_ids,
                  return_blob=False,
                  spectrum_only=False,
                  Emin=6e9):
        """
        return the chi2 for the fitted fractions in a format need by the emcee module
        """
        lst_models = self.compute_models(source_params, particle_ids)

        optimizer = UHECROptimizer(
            lst_models,
            self.spectrum,
            self.xmax,
            Emin=Emin,
            ncoids=particle_ids)
        minres = optimizer.fit_data_minuit(spectrum_only=spectrum_only)

        # return either only the chi2 for MCMC, or also the computation result
        # if the computation result is returned, it will be saved in the MCMC chain
        if return_blob:
            # return minres.fval, (minres, optimizer.lst_res)
            return minres.fval, (self.extract_minuit(minres),
                                 [res.to_dict() for res in optimizer.lst_res])
        else:
            return minres.fval

    def __call__(self, params, pids):
        return self.lnprob_mc(params, pids)

    def run_mcmc(self,
                 params,
                 pids,
                 nwalkers=100,
                 nsamples=100,
                 mpi=False,
                 threads=1):
        """
        Runs an MCMC chain
        """
        # Setup the pool, to map lnprop
        import schwimmbad
        import emcee

        pool = schwimmbad.choose_pool(mpi=mpi, processes=threads)

        ndim = len(params)
        pos0 = [
            params + params * np.random.randn(ndim) * 0.01
            for i in range(nwalkers)
        ]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self, args=(pids, ), pool=pool)
        sampler.run_mcmc(pos0, nsamples)

        pool.close()
        return sampler.chain