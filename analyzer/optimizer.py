import numpy as np
from xmax import XmaxSimple


class UHECROptimizer(object):
    EPOSLHC = XmaxSimple(model=XmaxSimple.EPOSLHC)

    def __init__(self, solver, spectrum, Xmax, Emin=6e9, fractions=None):
        self.Emin = Emin

        # spectral data
        self.spectrum = spectrum
        self.e_spectrum = spectrum['energy'].value

        # Xmax data
        self.Xmax = Xmax
        self.e_xmax = Xmax['energy'].value

        if type(solver) is list:
            self.lst_res = [s.res for s in solver]

            if fractions is None:
                fractions = [1. for s in solver]

            self.compute_combined_result(fractions)
        else:
            self.lst_res = None
            self._extract_solver_result(solver.res)

    def compute_combined_result(self, fractions):
        if self.lst_res is None and fractions is not None:
            raise Exception(
                'Provided fraction but no list of results to weight!')
        elif len(self.lst_res) != len(fractions):
            raise Exception(
                'Number of fractions ({:}) not equal to number of precomputed results ({:})'.
                format(
                    len(fractions),
                    len(self.lst_res),
                ))

        result_comb = reduce(
            lambda x, y: x + y,
            [res * frac for res, frac in zip(self.lst_res, fractions)])
        self._extract_solver_result(result_comb)

    def _extract_solver_result(self, res):
        self.res_spectrum = res.get_solution_group(
            [el for el in res.known_species if el >= 100],
            egrid=self.e_spectrum)[1]

        self.e_xmax, mean_lnA, sigma_lnA = res.get_lnA(
            [el for el in res.known_species if el >= 100], egrid=self.e_xmax)
        self.res_xmax = self.EPOSLHC.get_mean_Xmax(mean_lnA, self.e_xmax)
        self.var_xmax, _ = np.sqrt(
            self.EPOSLHC.get_sigma2_Xmax(mean_lnA, sigma_lnA**2, self.e_xmax))

    def get_chi2_spectrum(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.compute_combined_result(fractions)

        sl = np.where(self.e_spectrum > self.Emin)

        delta = norm * self.res_spectrum[sl] - self.spectrum['spectrum'].value[sl]
        error = np.where(
            norm * self.res_spectrum[sl] > self.spectrum['spectrum'].value[sl],
            self.spectrum['upper_err'][sl], self.spectrum['lower_err'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_Xmax(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.compute_combined_result(fractions)

        sl = np.where(self.e_xmax > self.Emin)

        delta = self.res_xmax[sl] - self.Xmax['Xmax'].value[sl]
        error = np.where(
            norm * self.res_xmax[sl] > self.Xmax['Xmax'].value[sl],
            self.Xmax['statXmax'][sl], self.Xmax['statXmax'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_VarXmax(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.compute_combined_result(fractions)

        sl = np.where(self.e_xmax > self.Emin)

        delta = self.var_xmax[sl] - self.Xmax['XRMS'].value[sl]
        error = np.where(
            norm * self.var_xmax[sl] > self.Xmax['XRMS'].value[sl],
            self.Xmax['statXRMS'][sl], self.Xmax['statXRMS'][sl])

        return np.sum((delta / error)**2)

    def get_chi2_total(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.compute_combined_result(fractions)

        return sum([
            self.get_chi2_spectrum(norm),
            self.get_chi2_Xmax(norm),
            self.get_chi2_VarXmax(norm)
        ])

    def fit_data(self, spectrum_only=False):
        from scipy.optimize import minimize

        if spectrum_only:

            def chi2(vec):
                return self.get_chi2_spectrum(vec[0], fractions=vec[1:])
        else:

            def chi2(vec):
                return self.get_chi2_total(vec[0], fractions=vec[1:])

        def fraction_constrain(vec):
            return np.sum(vec[1:]) - 100.

        cons = {'type': 'eq', 'fun': fraction_constrain}
        bounds = [(0., None) for val in range(len(self.lst_res) + 1)]
        init_guess = [1e4] + [1. for val in range(len(self.lst_res))]

        res = minimize(chi2, init_guess, bounds=bounds, constraints=cons)
        return res


class UHECRWalker(object):
    def __init__(self, prince_run, spectrum, xmax):
        self.prince_run = prince_run
        self.spectrum = spectrum
        self.xmax = xmax

    def compute_models(self, source_params, particle_ids):
        """
        Compute the results corresponding to source_params for each particle id individually and return a list
        """
        from prince.solvers import UHECRPropagationSolver
        from prince.cr_sources import AugerFitSource

        lst_models = []
        rmax, gamma = source_params
        for ncoid in particle_ids:

            solver = UHECRPropagationSolver(1., 0., self.prince_run)

            solver.add_source_class(
                AugerFitSource(
                    self.prince_run,
                    ncoid=ncoid,
                    rmax=rmax,
                    spectral_index=gamma,
                    norm=1e-50))
            solver.set_initial_condition()
            solver.solve(dz=1e-3, verbose=False, full_reset=False)

            lst_models.append(solver)

        # return the results only
        return lst_models

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
            lst_models, self.spectrum, self.xmax, Emin=Emin)
        optres = optimizer.fit_data(spectrum_only)

        # return either only the chi2 for MCMC, or also the computation result
        # if the computiation result is returned, it will be saved in the MCMC chain
        if return_blob:
            return optres.fun, (optres, optimizer.lst_res)
        else:
            return optres.fun

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
        # def lnprob(source_params, pids, prince_run):
        #     walker = UHECRWalker(prince_run, self.spectrum, self.xmax)
        #     res = walker.lnprob_mc(source_params,pids,return_blob=True,spectrum_only=True,Emin=6e9)
        #     return res

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