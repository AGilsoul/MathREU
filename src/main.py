import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

# HOW LONG TO TAKE IT FOR, HOW LONG TILL ERADICATED???
# USE RK45 INSTEAD OF EULER
# TAKE 5 DRUGS IN REGOES, SAY COURSE OF DRUG IS X NUMBER OF DAYS (LIKE 7, SO 21 DOSES TOTAL)
# FOR EACH, PERFECT PATIENT AFTER 7 DAYS, BACTERIAL DENSITY
# ERADICATED WHEN GOES 2 ORDERS OF MAGNITUDE ABOVE FINAL DENSITY
# THEN ADD MISSED DOSES (PROBABILITY OF 1-P OF MISSING DOSE, Q) FOR SINGLE DOSING
# IF DOSE MISSED IN SINGLE DOSE, ADD TO THE END
# PLOT Q AGAINST FINAL DENSITY
# DO FOR SINGLE AND DOUBLE DOSE


threshold = 1e-6


class Antibiotic:
    def __init__(self, psi_max, psi_min, k, MIC, zMIC, a0_MIC_ratio=5, name='Parameters'):
        self.psi_max = psi_max
        self.psi_min = psi_min
        self.k = k
        self.MIC = MIC
        self.zMIC = zMIC
        self.a0 = zMIC * a0_MIC_ratio
        self.last_a0 = self.a0
        self.name = name
        return


class Regimen:
    def __init__(self, time_range, antibiotic: Antibiotic, num_doses=21, dose_period=8, p=1.0, double_dose=False):
        self.last_timestep = 0
        self.time_since_dose = 0
        self.last_dose_missed = False
        self.took_dose = False
        self.num_doses = num_doses
        self.dose_period = dose_period
        self.time_range = time_range
        self.double_dose = double_dose
        self.p = p
        self.ab = [self.ab_conc(t, antibiotic) for t in time_range]

    def get_delta(self, MIC, a0):
        delta = (1 / self.dose_period) * np.log(0.5 * MIC / a0)
        return delta

    # Take dose IF
    # delta is 0 or difference between closest planned dose time and dose delta is
    def take_dose(self, t: float):
        # dose_delta is time since last dose taken
        factor = math.floor(t / self.dose_period)
        planned_dose = factor * self.dose_period
        if planned_dose + self.dose_period < t:
            planned_dose += self.dose_period

        good_slot = t >= planned_dose > self.last_timestep
        return good_slot and self.num_doses > 0 and not self.took_dose

    def update(self, t: float, dose_taken: bool):
        if dose_taken:
            print(f'took dose, setting dose missed to {not dose_taken}\n')
            self.time_since_dose = 0
            self.took_dose = True
        else:
            self.time_since_dose += t - self.last_timestep
            self.took_dose = False
        self.last_timestep = t

    # antibiotic concentration at time t (hours)
    def ab_conc(self, t: float, antibiotic: Antibiotic):
        return_val = 0
        # get half life of antibiotic
        delta = self.get_delta(antibiotic.MIC, antibiotic.a0)
        # get time since last dose THIS IS WRONG FIX THIS PLEASE
        # last_dose_delta = self.time_since_dose % self.dose_period

        # determine if a dose should be taken
        # INCORPORATE RANDOM MISSED DOSE
        if self.take_dose(t):
            rand_num = random.uniform(0, 1)
            print(f'dose time: {t}, last timestep: {self.last_timestep}')
            if rand_num < self.p:
                # if True:
                print(f'dose administered at t={t}, last dose missed: {self.last_dose_missed}')
                if self.last_dose_missed and self.double_dose:
                    print(f'double dosing')
                    antibiotic.last_a0 = 2 * antibiotic.a0
                else:
                    print(f'single dosing')
                    antibiotic.last_a0 = antibiotic.a0
                self.update(t, True)
                self.last_dose_missed = False
                return_val = antibiotic.last_a0
                self.num_doses -= 1
            else:
                print(f'dose missed at t={t}')
                self.update(t, False)
                self.last_dose_missed = True
                return_val = antibiotic.last_a0 * np.exp(delta * self.time_since_dose)
        else:
            self.update(t, False)
            return_val = antibiotic.last_a0 * np.exp(delta * self.time_since_dose)
        return return_val


class Simulation:
    def __init__(self):
        self.t_range = []
        self.population = []
        self.ab = []
        self.time_steps = []

    def show_sim(self, antibiotic, regimen):
        fig, ax = plt.subplots(1, 2)
        if regimen.double_dose:
            title = f'Double Dose {antibiotic.name}'
        else:
            title = f'Single Dose {antibiotic.name}'
        plt.title(title)
        ax[0].plot(self.t_range, self.population)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Bacterial Density')
        ax[0].set_xlabel('Time (hrs)')

        base_mic = [antibiotic.MIC for _ in self.t_range]
        base_zmic = [antibiotic.zMIC for _ in self.t_range]
        ax[1].plot(self.t_range, self.ab)
        ax[1].plot(self.t_range, base_mic, label='MIC')
        ax[1].plot(self.t_range, base_zmic, label='zMIC')
        ax[1].set_ylabel('Antibiotic Concentration')
        ax[1].set_xlabel('Time (hrs)')
        ax[1].legend(loc='upper right')
        plt.show()


class Euler(Simulation):
    def __init__(self, X_0, deriv, antibiotic: Antibiotic, regimen: Regimen):
        super().__init__()
        self.antibiotic = antibiotic
        self.regimen = regimen
        dt = num_hours / 10**k
        for i in range(0, len(regimen.time_range)):
            if i == 0:
                cur_X = X_0
            else:
                cur_X = self.population[-1]
            dx, a = deriv(regimen.time_range[i], cur_X, antibiotic, regimen)
            self.population.append(cur_X + dx * dt)
            self.ab.append(a)

        self.population = np.array(self.population)
        self.ab = np.array(self.ab)
        return


class RK45(Simulation):
    def __init__(self, X_0, deriv, antibiotic: Antibiotic, regimen: Regimen):
        super().__init__()
        t_span = (0, num_hours)
        self.t_range = regimen.time_range
        ab_vals = regimen.ab
        solX = solve_ivp(lambda t, x: deriv(t, x, antibiotic, regimen), t_span, [X_0], t_eval=regimen.time_range, dense_output=True, method='RK45')
        z = solX.sol(self.t_range)[0]
        self.population = z
        self.ab = ab_vals
        return


k = 5
days = 10
num_hours = 24 * days  # 7 days in hours
test_range = np.linspace(0, num_hours, 10**k)
set_missed_doses = [2]
q = 0.1
p = 1-q


Ciprofloxacin = Antibiotic(0.88, -6.5, 1.1, 0.017, 0.03, name='Ciprofloxacin')
Ampicillin = Antibiotic(0.75, -4.0, 0.75, 3.4, 8.0, name='Ampicillin')
Rifampin = Antibiotic(0.7, -4.3, 2.5, 12.0, 8.0, name='Rifampin')
Streptomycin = Antibiotic(0.89, -8.8, 1.9, 18.5, 32.0, name='Streptomycin')
Tetracycline = Antibiotic(0.81, -8.1, 0.61, 0.67, 1.0, name='Tetracycline')

# base_regimen = Regimen(test_range, Ampicillin)
# single_regimen = Model.Regimen(test_range, num_hours=num_hours, doses_missed=set_missed_doses, double_dose=True)
single_regimen = Regimen(test_range, Ampicillin, double_dose=False, p=p)
double_regimen = Regimen(test_range, Ampicillin, double_dose=True, p=p)


# growth rate of bacterial population
def psi(ab_conc: float, antibiotic: Antibiotic):
    numerator = (antibiotic.psi_max - antibiotic.psi_min) * (ab_conc/antibiotic.zMIC)**antibiotic.k
    denominator = (ab_conc/antibiotic.zMIC)**antibiotic.k - (antibiotic.psi_min/antibiotic.psi_max)
    return antibiotic.psi_max - (numerator / denominator)


# change in density, X, of a bacterial population
def dX_dt(t: float, X: float, antibiotic: Antibiotic, regimen: Regimen):
    dx = 0
    if X >= threshold:
        ab = np.interp(t, regimen.time_range, regimen.ab)
        # print(f't: {t}, ab: {ab}')
        dx = np.log(10) * psi(ab, antibiotic) * X
    return dx


def main():
    # basic_euler(10**6, Ciprofloxacin)
    # basic_euler(10**6, Ampicillin)
    # basic_euler(10**6, Rifampin)
    # basic_euler(10**6, Streptomycin)
    # basic_euler(10**6, Tetracycline)
    # sim = RK45(10**6, dX_dt, Ampicillin, single_regimen)
    sim = RK45(10**6, dX_dt, Ampicillin, double_regimen)
    # sim = Euler(10**6, dX_dt, Tetracycline, single_regimen)
    sim.show_sim(Ampicillin, single_regimen)
    return


if __name__ == '__main__':
    main()

