import numpy as np
import matplotlib.pyplot as plt


class Model:
    class Bacteria:
        # psi_max: max growth rate in absence of antibiotics
        def __init__(self, psi_max):
            self.psi_max = psi_max
            return

    class ParamSet:
        def __init__(self, E_max, k, MIC, zMIC, dose_frequency=0.125, a0_MIC_ratio=5):
            self.E_max = E_max
            self.k = k
            self.MIC = MIC
            self.zMIC = zMIC
            self.a0 = zMIC * a0_MIC_ratio
            self.dose_frequency = dose_frequency
            return


# growth rate of bacterial population
def psi(ab_conc: float, bacteria: Model.Bacteria, params: Model.ParamSet):
    psi_min = bacteria.psi_max - params.E_max
    return bacteria.psi_max - (bacteria.psi_max - psi_min) * (ab_conc/params.zMIC)**params.k / ((ab_conc/params.zMIC)**params.k - (psi_min/bacteria.psi_max))


# change in density, X, of a bacterial population
def dX_dt(t: float, bacteria: Model.Bacteria, params: Model.ParamSet):

    return


# antibiotic concentration at time t (hours)
def a(t: float, params: Model.ParamSet):
    delta = params.dose_frequency * np.log(0.5 * params.MIC / params.a0)
    if t % (1 / params.dose_frequency) == 0:
        # break
        return


def main():

    return


if __name__ == '__main__':
    main()