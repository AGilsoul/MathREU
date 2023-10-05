import numpy as np
import matplotlib.pyplot as plt


class Model:
    class Antibiotic:
        def __init__(self, psi_max, psi_min, k, MIC, zMIC, a0_MIC_ratio=5, name='Parameters'):
            self.psi_max = psi_max
            self.psi_min = psi_min
            self.k = k
            self.MIC = MIC
            self.zMIC = zMIC
            self.a0 = zMIC * a0_MIC_ratio
            self.name = name
            return

    class Regimen:
        doses = []
        cur_dose = 0
        time_since_dose = 0

        def __init__(self, time_range, dose_frequency=0.125, num_hours=24, doses_missed=[], double_dose=False):
            self.time_range = time_range
            self.dose_frequency = dose_frequency
            self.num_hours = num_hours
            self.doses_missed = doses_missed
            self.double_dose = double_dose
            self.num_doses = int(num_hours * dose_frequency)
            self.doses = [1 if i not in doses_missed else 0 for i in range(self.num_doses)]


class Simulation:
    antibiotic = Model.Antibiotic
    regimen = Model.Regimen
    population = []
    ab = []
    time_steps = []

    def show_sim(self):
        fig, ax = plt.subplots(1, 2)
        plt.title(self.antibiotic.name)
        ax[0].plot(self.regimen.time_range, self.population)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Bacterial Density')
        ax[0].set_xlabel('Time (hrs)')

        base_mic = [self.antibiotic.MIC for _ in self.regimen.time_range]
        base_zmic = [self.antibiotic.zMIC for _ in self.regimen.time_range]
        ax[1].plot(self.regimen.time_range, self.ab)
        ax[1].plot(self.regimen.time_range, base_mic, label='MIC')
        ax[1].plot(self.regimen.time_range, base_zmic, label='zMIC')
        ax[1].set_ylabel('Antibiotic Concentration')
        ax[1].set_xlabel('Time (hrs)')
        ax[1].legend()
        plt.show()


class Euler(Simulation):
    def __init__(self, X_0, deriv, antibiotic: Model.Antibiotic, regimen: Model.Regimen):
        super().__init__()
        self.antibiotic = antibiotic
        self.regimen = regimen
        dt = num_hours / 10**k
        print(f'dt: {dt}')
        for i in range(0, len(regimen.time_range)):
            if i == 0:
                cur_X = X_0
            else:
                dt = regimen.time_range[i] - regimen.time_range[i-1]
                cur_X = self.population[-1]
            dx, a = deriv(regimen.time_range[i], dt, cur_X, antibiotic, regimen)
            self.population.append(cur_X + dx * dt)
            self.ab.append(a)

        self.population = np.array(self.population)
        self.ab = np.array(self.ab)
        return


k = 3
num_hours = 240
test_range = np.linspace(0, num_hours, 10**k)
set_missed_doses = [1, 7, 9, 11, 23]

base_regimen = Model.Regimen(test_range)
single_regimen = Model.Regimen(test_range, num_hours=num_hours, doses_missed=set_missed_doses)

Ciprofloxacin = Model.Antibiotic(0.88, -6.5, 1.1, 0.017, 0.03, name='Ciprofloxacin')
Ampicillin = Model.Antibiotic(0.75, -4.0, 0.75, 3.4, 8.0, name='Ampicillin')
Rifampin = Model.Antibiotic(0.7, -4.3, 2.5, 12.0, 8.0, name='Rifampin')
Streptomycin = Model.Antibiotic(0.89, -8.8, 1.9, 18.5, 32.0, name='Streptomycin')
Tetracycline = Model.Antibiotic(0.81, -8.1, 0.61, 0.67, 1.0, name='Tetracycline')


# growth rate of bacterial population
def psi(ab_conc: float, antibiotic: Model.Antibiotic):
    numerator = (antibiotic.psi_max - antibiotic.psi_min) * (ab_conc/antibiotic.zMIC)**antibiotic.k
    denominator = (ab_conc/antibiotic.zMIC)**antibiotic.k - (antibiotic.psi_min/antibiotic.psi_max)
    return antibiotic.psi_max - (numerator / denominator)


# change in density, X, of a bacterial population
def dX_dt(t: float, dt: float, X: float, antibiotic: Model.Antibiotic, regimen: Model.Regimen):
    ab = ab_conc(t, dt, antibiotic, regimen)
    return np.log(10) * psi(ab, antibiotic) * X, ab


# antibiotic concentration at time t (hours)
def ab_conc(t: float, dt: float, antibiotic: Model.Antibiotic, regimen: Model.Regimen):
    delta = regimen.dose_frequency * np.log(0.5*antibiotic.MIC/antibiotic.a0)

    dose_period = (1 / regimen.dose_frequency)

    last_dose_delta = round(regimen.time_since_dose, k) % dose_period

    if (last_dose_delta == 0 or abs(dose_period - last_dose_delta) < dt) and regimen.cur_dose < len(regimen.doses):
        regimen.cur_dose += 1
        if regimen.doses[regimen.cur_dose - 1] == 1:
            print(f'dose {regimen.cur_dose} given at {t}')
            regimen.time_since_dose = dt
            return antibiotic.a0
        else:
            print(f'dose {regimen.cur_dose} missed at  {t}')
    regimen.time_since_dose += dt
    return antibiotic.a0 * np.exp(delta * regimen.time_since_dose)


def main():
    # basic_euler(10**6, Ciprofloxacin)
    # basic_euler(10**6, Ampicillin)
    # basic_euler(10**6, Rifampin)
    # basic_euler(10**6, Streptomycin)
    # basic_euler(10**6, Tetracycline)
    sim = Euler(10**6, dX_dt, Ampicillin, single_regimen)
    sim.show_sim()
    return


if __name__ == '__main__':
    main()

