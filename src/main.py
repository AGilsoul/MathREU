import numpy as np
import matplotlib.pyplot as plt


class Model:
    class ParamSet:
        doses_given = 0

        def __init__(self, psi_max, psi_min, k, MIC, zMIC, time_range, dose_frequency=0.125, a0_MIC_ratio=5, num_hours=24, doses_missed=[], double_dose=False, name='Parameters'):
            self.psi_max = psi_max
            self.psi_min = psi_min
            self.k = k
            self.MIC = MIC
            self.zMIC = zMIC
            self.time_range = time_range
            self.a0 = zMIC * a0_MIC_ratio
            self.dose_frequency = dose_frequency
            self.num_hours = num_hours
            self.doses_missed = doses_missed
            self.double_dose = double_dose
            self.name = name
            return


class Simulation:
    params = Model.ParamSet
    population = []
    ab = []
    time_steps = []

    def show_sim(self):
        fig, ax = plt.subplots(1, 2)
        plt.title(self.params.name)
        ax[0].plot(self.params.time_range, self.population)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Bacterial Density')
        ax[0].set_xlabel('Time (hrs)')

        base_mic = [self.params.MIC for _ in self.params.time_range]
        base_zmic = [self.params.zMIC for _ in self.params.time_range]
        ax[1].plot(self.params.time_range, self.ab)
        ax[1].plot(self.params.time_range, base_mic, label='MIC')
        ax[1].plot(self.params.time_range, base_zmic, label='zMIC')
        ax[1].set_ylabel('Antibiotic Concentration')
        ax[1].set_xlabel('Time (hrs)')
        ax[1].legend()
        plt.show()


class Euler(Simulation):
    def __init__(self, X_0, deriv, params: Model.ParamSet):
        super().__init__()
        self.params = params
        dt = params.time_range[1] - params.time_range[0]
        for i in range(0, len(params.time_range)):
            if i == 0:
                cur_X = X_0
            else:
                cur_X = self.population[-1]
            dx, a = deriv(params.time_range[i], cur_X, params)
            self.population.append(cur_X + dx * dt)
            self.ab.append(a)

        self.population = np.array(self.population)
        self.ab = np.array(self.ab)
        return


test_range = np.linspace(0, 24, 1_000)
Ciprofloxacin = Model.ParamSet(0.88, -6.5, 1.1, 0.017, 0.03, test_range, name='Ciprofloxacin')
Ampicillin = Model.ParamSet(0.75, -4.0, 0.75, 3.4, 8.0, test_range, name='Ampicillin')
Rifampin = Model.ParamSet(0.7, -4.3, 2.5, 12.0, 8.0, test_range, name='Rifampin')
Streptomycin = Model.ParamSet(0.89, -8.8, 1.9, 18.5, 32.0, test_range, name='Streptomycin')
Tetracycline = Model.ParamSet(0.81, -8.1, 0.61, 0.67, 1.0, test_range, name='Tetracycline')


# growth rate of bacterial population
def psi(ab_conc: float, params: Model.ParamSet):
    numerator = (params.psi_max - params.psi_min) * (ab_conc/params.zMIC)**params.k
    denominator = (ab_conc/params.zMIC)**params.k - (params.psi_min/params.psi_max)
    return params.psi_max - (numerator / denominator)


# change in density, X, of a bacterial population
def dX_dt(t: float, X: float, params: Model.ParamSet):
    ab = ab_conc(t, params)
    return np.log(10) * psi(ab, params) * X, ab


# antibiotic concentration at time t (hours)
def ab_conc(t: float, params: Model.ParamSet):
    time_since_dose = t % (1 / params.dose_frequency)
    if time_since_dose == 0:
        params.doses_given += 1
        if params.doses_given not in params.doses_missed:
            return params.a0
    delta = params.dose_frequency * np.log(0.5*params.MIC/params.a0)
    return params.a0 * np.exp(delta * time_since_dose)


def main():
    # basic_euler(10**6, Ciprofloxacin)
    # basic_euler(10**6, Ampicillin)
    # basic_euler(10**6, Rifampin)
    # basic_euler(10**6, Streptomycin)
    # basic_euler(10**6, Tetracycline)
    sim = Euler(10**6, dX_dt, Ampicillin)
    sim.show_sim()
    return


if __name__ == '__main__':
    main()

