import math
from params import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

# READ UP ON TOXICITY!!!!!!!!!!!!!
# HOW LONG TO TAKE IT FOR, HOW LONG TILL ERADICATED???
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
        # self.ab = [self.ab_conc(t, antibiotic) for t in time_range]

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

        base_zmic = [PD.zMIC for _ in self.t_range]
        ax[1].plot(self.t_range, self.ab)
        ax[1].plot(self.t_range, base_zmic, label='zMIC')
        ax[1].set_ylabel('Antibiotic Concentration')
        ax[1].set_xlabel('Time (hrs)')
        ax[1].legend(loc='upper right')
        plt.show()


Ampicillin = Antibiotic(0.75, -4.0, 0.75, 3.4, 8.0, name='Ampicillin')

single_regimen = Regimen(test_range, Ampicillin, double_dose=False, p=p)
double_regimen = Regimen(test_range, Ampicillin, double_dose=True, p=p)


def f(b: float, c: float):
    return PD.r * (1 - (b / PD.K)) - (PD.r + PD.d) * ((c/PD.zMIC)**PD.k) / ((PD.d/PD.r) + (c/PD.zMIC)**PD.k)


def dc_dt(c: float):
    res = -PK.k_e * c
    return res


def db_dt(b: float, c: float):
    return f(b, c) * b


def d_system(t: float, ic):
    b = ic[0]
    c = ic[1]
    d1 = db_dt(b, c)
    d2 = dc_dt(c)
    return [d1, d2]


def take_dose(t: float):
    return True


global AUC_times

# READ CAROL PAPER, FIGUREOUT NONMEM
# Explore fT>MIC(t) and fAUC:MIC(t)
# BREAK INTO SMALLER TIME STEPS WITH ODE FOR A
# da/dt = - delta * a
class RK45(Simulation):
    def __init__(self, X_0, deriv, antibiotic: Antibiotic, regimen: Regimen):
        super().__init__()
        num_intervals = int(num_hours / regimen.dose_period)
        # num_intervals = 4
        b = np.array([PD.b_0])
        c = np.array([PK.D])
        t_interval = np.array([0])
        for i in range(num_intervals):
            # solve_ivp for system here
            eval_points = np.linspace(i * regimen.dose_period, (i+1) * regimen.dose_period, 1000)
            if take_dose(i):
                ic = [b[-1], PK.D]
            else:
                ic = [b[-1], c[-1]]
            print(f'IC: {ic}')
            t_span = (i * regimen.dose_period, (i+1 * regimen.dose_period))
            sol = solve_ivp(d_system, t_span, ic, dense_output=True, method='RK45')
            # print(sol)
            z = sol.sol(eval_points)
            b_vals = z[0]
            c_vals = z[1]
            b = np.append(b, b_vals)
            c = np.append(c, c_vals)
            print(b)
            # print(f'c: {c}')
            # print(f't_interval: {eval_points}')
            t_interval = np.append(t_interval, eval_points)
        self.population = b
        self.ab = c
        self.t_range = t_interval
        return



def main():
    # basic_euler(10**6, Ciprofloxacin)
    # basic_euler(10**6, Ampicillin)
    # basic_euler(10**6, Rifampin)
    # basic_euler(10**6, Streptomycin)
    # basic_euler(10**6, Tetracycline)
    # sim = RK45(10**6, dX_dt, Ampicillin, single_regimen)
    sim = RK45(10**6, db_dt, Ampicillin, double_regimen)
    sim.show_sim(Ampicillin, double_regimen)
    # sim = Euler(10**6, dX_dt, Tetracycline, single_regimen)
    # sim.show_sim(Ampicillin, double_regimen)
    return


if __name__ == '__main__':
    main()

