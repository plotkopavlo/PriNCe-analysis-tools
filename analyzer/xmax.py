import numpy as np


def convert_to_namedtuple(dictionary, name='GenericNamedTuple'):
    """Converts a dictionary to a named tuple."""
    from collections import namedtuple
    return namedtuple(name, dictionary.keys())(**dictionary)


class XmaxSimple(object):

    EPOS = {
        'X0': 809.7,  # +- 0.3  # g cm**-2
        'D': 62.2,  # +- 0.5  # g cm**-2
        'xi': 0.78,  # +- 0.24 # g cm**-2
        'delta': 0.08,  # +- 0.21 # g cm**-2
        'p0': 3279,  # +- 51   # g**2 cm**-4
        'p1': -47,  # +- 66   # g**2 cm**-4
        'p2': 228,  # +- 108  # g**2 cm**-4
        'a0': -0.461,  # +- 0.006
        'a1': -0.0041,  # +- 0.0016
        'b': 0.059,  # +- 0.002
        'E0': 1e10,  # GeV
    }
    EPOS = convert_to_namedtuple(EPOS, name='EPOS')

    Sybill = {
        'X0': 795.1,  # +- 0.3  # g cm**-2
        'D': 57.7,  # +- 0.5  # g cm**-2
        'xi': -0.04,  # +- 0.24 # g cm**-2
        'delta': -0.04,  # +- 0.21 # g cm**-2
        'p0': 2785,  # +- 46   # g**2 cm**-4
        'p1': -364,  # +- 58   # g**2 cm**-4
        'p2': 152,  # +- 93   # g**2 cm**-4
        'a0': -0.368,  # +- 0.008
        'a1': -0.0049,  # +- 0.0023
        'b': 0.039,  # +- 0.002
        'E0': 1e10,  # GeV
    }
    Sybill = convert_to_namedtuple(Sybill, name='Sybill')

    QGSJet01 = {
        'X0': 774.2,  # +- 0.3  # g cm**-2
        'D': 49.7,  # +- 0.5  # g cm**-2
        'xi': -0.30,  # +- 0.24 # g cm**-2
        'delta': 1.92,  # +- 0.21 # g cm**-2
        'p0': 3852,  # +- 55   # g**2 cm**-4
        'p1': -274,  # +- 70   # g**2 cm**-4
        'p2': 169,  # +- 116  # g**2 cm**-4
        'a0': -0.451,  # +- 0.006
        'a1': -0.0020,  # +- 0.0016
        'b': 0.057,  # +- 0.001
        'E0': 1e10,  # GeV
    }
    QGSJet01 = convert_to_namedtuple(QGSJet01, name='QGSJet01')

    QGSJetII = {
        'X0': 781.8,  # +- 0.3  # g cm**-2
        'D': 45.8,  # +- 0.5  # g cm**-2
        'xi': -1.13,  # +- 0.24 # g cm**-2
        'delta': 1.71,  # +- 0.21 # g cm**-2
        'p0': 3163,  # +- 49   # g**2 cm**-4
        'p1': -237,  # +- 61   # g**2 cm**-4
        'p2': 60,  # +- 100  # g**2 cm**-4
        'a0': -0.386,  # +- 0.007
        'a1': -0.0006,  # +- 0.0021
        'b': 0.043,  # +- 0.002
        'E0': 1e10,  # GeV
    }
    QGSJetII = convert_to_namedtuple(QGSJetII, name='QGSJetII')

    # -------------------------------------------------------------
    # Note JH: The folling 3 parameter sets are Auger internal
    #          Check for permission before using in publications
    # -------------------------------------------------------------

    Sibyll23 = {
        'X0': 824.3,  # +- 0.2  # g cm**-2
        'D': 58.4,  # +- 0.2  # g cm**-2
        'xi': -0.38,  # +- 0.08 # g cm**-2
        'delta': 0.59,  # +- 0.06 # g cm**-2
        'p0': 3810,  # +- 27  # g**2 cm**-4
        'p1': -405,  # +- 30   # g**2 cm**-4
        'p2': 125,  # +- 23  # g**2 cm**-4
        'a0': -0.406,  # +- 0.003
        'a1': -0.0016,  # +- 0.0006
        'b': 0.047,  # +- 0.001
        'E0': 1e10,  # GeV
    }
    Sibyll23 = convert_to_namedtuple(Sibyll23, name='Sibyll23')

    EPOSLHC = {
        'X0': 806.0,  # +- 0.3  # g cm**-2
        'D': 56.3,  # +- 0.2  # g cm**-2
        'xi': 0.35,  # +- 0.12 # g cm**-2
        'delta': 1.04,  # +- 0.10 # g cm**-2
        'p0': 3269,  # +- 39  # g**2 cm**-4
        'p1': -305,  # +- 42   # g**2 cm**-4
        'p2': 124,  # +- 32  # g**2 cm**-4
        'a0': -0.460,  # +- 0.005
        'a1': -0.0006,  # +- 0.0007
        'b': 0.058,  # +- 0.001
        'E0': 1e10,  # GeV
    }
    EPOSLHC = convert_to_namedtuple(EPOSLHC, name='EPOSLHC')

    QGSJetII04 = {
        'X0': 790.1,  # +- 0.3  # g cm**-2
        'D': 54.2,  # +- 0.2  # g cm**-2
        'xi': -0.42,  # +- 0.12 # g cm**-2
        'delta': 0.69,  # +- 0.10 # g cm**-2
        'p0': 3688,  # +- 41  # g**2 cm**-4
        'p1': -428,  # +- 42   # g**2 cm**-4
        'p2': 51,  # +- 32  # g**2 cm**-4
        'a0': -0.394,  # +- 0.005
        'a1': 0.0013,  # +- 0.0009
        'b': 0.045,  # +- 0.001
        'E0': 1e10,  # GeV
    }
    QGSJetII04 = convert_to_namedtuple(QGSJetII04, name='EPOSQGSJetII04LHC')

    def __init__(self, model=None):
        if model is None:
            self.model = self.EPOS
        else:
            self.model = model

    def get_mean_Xmax(self, mean_lnA, E):
        """Returns the mean X_max at energy E for mass number A
        """
        m = self.model

        # following ref eq. (2.1)
        Xmax_proton = m.X0 + m.D * np.log10(E / m.E0)
        # following ref eq. (2.5)
        fE = m.xi - m.D / np.log(10) + m.delta * np.log10(E / m.E0)

        # following ref eq. (2.6)
        return Xmax_proton + fE * mean_lnA

    def get_sigma2_Xmax(self, mean_lnA, sigma2_lnA, E):
        """Returns the squared standard deviation of X_max at energy E for mass number A
        """
        mean_lnA2 = sigma2_lnA + mean_lnA**2
        m = self.model

        # following ref eq. (2.5)
        fE = m.xi - m.D / np.log(10) + m.delta * np.log10(E / m.E0)

        # following ref eq. (2.9)
        sigma2_proton = m.p0 + m.p1 * np.log10(E / m.E0) + m.p2 * np.log10(
            E / m.E0)**2
        a = m.a0 + m.a1 * np.log10(E / m.E0)

        # following ref eq. (2.11)
        mean_sigma2_sh = sigma2_proton * (1 + a * mean_lnA + m.b * mean_lnA2)

        # following ref eq. (2.7)
        return mean_sigma2_sh + fE**2 * sigma2_lnA, mean_sigma2_sh