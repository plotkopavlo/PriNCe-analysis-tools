import numpy as np
from xmax import XmaxSimple


class UHECROptimizer(object):

    EPOSLHC = XmaxSimple(model = XmaxSimple.EPOSLHC)

    def __init__(self, solver, spectrum, Xmax, Emin=6e9, fractions=None):
        self.Emin = Emin

        # spectral data
        self.spectrum = spectrum
        self.e_spectrum = spectrum.energy.value

        # Xmax data
        self.Xmax = Xmax
        self.e_xmax = Xmax.energy.value

        if type(solver) is list:
            self.lst_res = [s.res for s in solver]

            if fractions is None:
                fractions = [1. for res in solver]

            result_comb = reduce(lambda x,y: x + y,
                               [res*frac for res, frac in zip(self.lst_res, fractions)])

            self._extract_solver_result(result_comb)
        else:
            self.lst_res = None
            self._extract_solver_result(solver.res)

    def recompute(self, fractions):
        result_comb = reduce(lambda x,y: x + y, 
                           [res*frac for res, frac in zip(self.lst_res, fractions)])
        self._extract_solver_result(result_comb)

    def _extract_solver_result(self, res):
        self.res_spectrum = res.get_solution_group(
            [el for el in res.known_species if el >= 100], egrid=self.e_spectrum)[1]

        self.e_xmax, mean_lnA, sigma_lnA = res.get_lnA(
            [el for el in res.known_species if el >= 100], egrid=self.e_xmax)
        self.res_xmax = self.EPOSLHC.get_mean_Xmax(mean_lnA, self.e_xmax)
        self.var_xmax,_ = np.sqrt(self.EPOSLHC.get_sigma2_Xmax(mean_lnA, sigma_lnA**2, self.e_xmax))

    def get_chi2_spectrum(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.recompute(fractions)

        sl = np.where(self.e_spectrum > self.Emin)

        delta = norm * self.res_spectrum[sl] - self.spectrum.spectrum.value[sl]
        error = np.where(norm * self.res_spectrum[sl] > self.spectrum.spectrum.value[sl],
                         self.spectrum.upper_err[sl],
                         self.spectrum.lower_err[sl])

        return np.sum((delta / error)**2)

    def get_chi2_Xmax(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.recompute(fractions)

        sl = np.where(self.e_xmax > self.Emin)

        delta = self.res_xmax[sl] - self.Xmax.Xmax.value[sl]
        error = np.where(norm * self.res_xmax[sl] > self.Xmax.Xmax.value[sl],
                         self.Xmax.statXmax[sl],
                         self.Xmax.statXmax[sl])

        return np.sum((delta / error)**2)

    def get_chi2_VarXmax(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.recompute(fractions)

        sl = np.where(self.e_xmax > self.Emin)

        delta = self.var_xmax[sl] - self.Xmax.XRMS.value[sl]
        error = np.where(norm * self.var_xmax[sl] > self.Xmax.XRMS.value[sl],
                         self.Xmax.statXRMS[sl],
                         self.Xmax.statXRMS[sl])

        return np.sum((delta / error)**2)

    def get_chi2_total(self, norm, fractions=None):
        if self.lst_res is not None and fractions is not None:
            self.recompute(fractions)

        return (self.get_chi2_spectrum(norm)
                + self.get_chi2_Xmax(norm)
                + self.get_chi2_VarXmax(norm))

    def fit_data(self):
        from scipy.optimize import minimize

        def chi2(vec):
            self.get_chi2_total(vec[0], fractions=vec[1:])
        def fraction_constrain(vec):
            return np.sum(vec[1:]) - 100.   
        cons = {'type':'eq', 'fun': fraction_constrain}
        bounds = [(0., None) for val in [1e4, 0.,60.,30.,10.,0.]]

        res = minimize(chi2, [1e4,1.,1.,1.,1.,1.], bounds=bounds, constraints=cons)
        return res

# class Spectrum(object):

#     def __init__(self, solver, spectrum, Xmax, Emin=6e9):
#         self.Emin = Emin

#         # spectral data
#         self.spectrum = spectrum
#         self.e_spectrum = spectrum.energy.value

#         self.res_spectrum = solver.res.get_solution_group(
#             [el for el in solver.known_species if el >= 100], egrid=self.e_spectrum)[1]

#         # Xmax data
#         self.Xmax = Xmax
#         self.e_xmax = Xmax.energy.value

#         self.e_xmax, mean_lnA, sigma_lnA = solver.res.get_lnA(
#             [el for el in solver.known_species if el >= 100], egrid=self.e_xmax)
#         self.res_xmax = x_EPOSLHC.get_mean_Xmax(mean_lnA, self.e_xmax)
#         self.var_xmax, _ = np.sqrt(x_EPOSLHC.get_sigma2_Xmax(mean_lnA, sigma_lnA**2, self.e_xmax))

#     def get_chi2_spectrum(self, norm):
#         sl = np.where(self.e_spectrum > self.Emin)

#         delta = norm * self.res_spectrum[sl] - self.spectrum.spectrum.value[sl]
#         error = np.where(norm * self.res_spectrum[sl] > self.spectrum.spectrum.value[sl],
#                          self.spectrum.upper_err[sl],
#                          self.spectrum.lower_err[sl] )

#         return np.sum((delta / error)**2)

#     def get_chi2_Xmax(self, norm):
#         sl = np.where(self.e_xmax > self.Emin)

#         delta = self.res_xmax[sl] - self.Xmax.Xmax.value[sl]
#         error = np.where(norm * self.res_xmax[sl] > self.Xmax.Xmax.value[sl],
#                          self.Xmax.statXmax[sl],
#                          self.Xmax.statXmax[sl])

#         return np.sum((delta / error)**2)

#     def get_chi2_VarXmax(self, norm):
#         sl = np.where(self.e_xmax > self.Emin)

#         delta = self.var_xmax[sl] - self.Xmax.XRMS.value[sl]
#         error = np.where(norm * self.var_xmax[sl] > self.Xmax.XRMS.value[sl],
#                          self.Xmax.statXRMS[sl],
#                          self.Xmax.statXRMS[sl])

#         return np.sum((delta / error)**2)

#     def get_chi2_total(self, norm):

#         return (self.get_chi2_spectrum(norm)
#                 + self.get_chi2_Xmax(norm)
#                 + self.get_chi2_VarXmax(norm))