import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


tvka = 1.02  # first order absorption rate constant
tvVd = 27.7  # drug's volume of distribution in L
tvCl = 21.3  # drug clearance in L/hr
rate_elim = 0.27  # Cl/Vd
mic = 1  # mg/mL
weight = 70  # weight in kg
mic_mg = mic * weight # mic in mg


class TimeStep:
    def __init__(self, t, Agi, C):
        self.t = t
        self.C = C
        self.Agi = Agi


# dC/dt, rate of drug absorbed into the body based on amount of drug in the GI tract
def d_dt_system(t, IC, ka, Cl):
    Agi = IC[0]
    C = IC[1]
    return [-ka*Agi, ka*Agi - rate_elim * C]


def take_dose(hr: int):
    # if hr != 0 and hr % 16 == 0:
    #     return False
    return True


def dose_to_dosage(conc, weight):
    return conc / weight


def percent_t_above_mic(conc, weight, dt, total_time):
    dosage = dose_to_dosage(conc, weight)
    time_above = sum([1 * dt if x > mic else 0 for x in dosage])
    return (time_above / total_time) * 100


def NONMEM_Based(ka, weight, cl, age):
    sim_time = 48    # total simulation time in hours
    dose_amt = 1000  # dose amount in milligrams
    dose_interval = 8  # number of hours between doses
    time_steps = []
    for hr in range(sim_time):
        if hr == 0 and take_dose(hr):
            print(f'hour 0, dose given')
            IC = [dose_amt, 0]
        elif hr % dose_interval == 0 and take_dose(hr):
            print(f'hour {hr}, dose given')
            IC = [time_steps[-1].Agi + dose_amt, time_steps[-1].C]
        else:
            print(f'hour {hr}')
            IC = [time_steps[-1].Agi, time_steps[-1].C]
        eval_points = np.linspace(hr, hr + 1, 1000)
        t_span = (hr, hr + 1)
        sol = solve_ivp(lambda t, y: d_dt_system(t, y, ka, cl), t_span, IC, dense_output=True, method='RK45')
        z = sol.sol(eval_points)
        for i in range(len(eval_points)):
            time_steps.append(TimeStep(eval_points[i], z[0][i], z[1][i]))

    dt = eval_points[1] - eval_points[0]
    gi_conc = np.array([x.Agi for x in time_steps])
    ab_conc = np.array([x.C for x in time_steps])
    t = np.array([x.t for x in time_steps])
    mic_r = np.array([mic_mg for x in time_steps])
    p_time_above = percent_t_above_mic(ab_conc, weight, dt, sim_time)
    print(p_time_above)
    plt.plot(t, ab_conc, label='AMOX in compartment')
    plt.plot(t, gi_conc, label='AMOX in gi tract')
    plt.plot(t, mic_r, label='MIC')
    plt.legend()
    plt.show()
    return p_time_above


def main():
    NONMEM_Based(tvka, weight, tvCl, 1)
    num_subjects = 100
    for i in range(num_subjects):
        NONMEM_Based()


if __name__ == '__main__':
    main()
