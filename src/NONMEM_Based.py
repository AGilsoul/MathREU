import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


tvage = 45
tvka = 0.635  # first order absorption rate constant
tvkd = 1.300
tvVd = 27.7  # drug's volume of distribution in L
tvCl = 21.3  # drug clearance in L/hr
tvrate_elim = 0.27  # Cl/Vd
mic = 1  # mg/mL
tvweight = 70  # weight in kg
tvmic_mg = mic * tvweight # mic in mg

omega = [0.04, 0.036, 0.01, 0.04, 0.001, 0.025]
theta1 = 0.635
theta2 = 15.5
theta3 = 0.33
theta8 = 1.300


class TimeStep:
    def __init__(self, t, Agi, C):
        self.t = t
        self.C = C
        self.Agi = Agi


# dC/dt, rate of drug absorbed into the body based on amount of drug in the GI tract
def d_dt_system(t, IC, ka, kd, rate_elim):
    Agi = IC[0]
    C = IC[1]
    return [-ka*Agi, ka*Agi - rate_elim * kd * C]


def take_dose(hr: int):
    if hr == 120:
        return False

    return True


def dose_to_dosage(conc, weight):
    return conc / weight


def percent_t_above_mic(conc, weight, dt, total_time):
    dosage = dose_to_dosage(conc, weight)
    # print(f'conc: {conc}, dosage: {dosage}')
    time_above = sum([1 * dt if x > mic else 0 for x in dosage])
    return (time_above / total_time) * 100


# ASSUME CLCR IS SAME AS CL
def NONMEM_Based(ka, kd, weight, rate_elim):
    sim_time = 240    # total simulation time in hours
    dose_amt = 1000  # dose amount in milligrams
    dose_interval = 8  # number of hours between doses
    time_steps = []
    time_since_dose = 0
    delay = 0
    new_delay = True
    for hr in range(sim_time):
        if hr == 4 * 24:
            delay = 8
        if hr == 0:
            # print(f'hour 0, dose given')
            IC = [dose_amt, 0]
            time_since_dose = 0
        elif time_since_dose % (dose_interval + delay) == 0:
            # print(f'hour {hr}, dose given')
            IC = [time_steps[-1].Agi + dose_amt, time_steps[-1].C]
            time_since_dose = 0
            delay = 0
        else:
            # print(f'hour {hr}')
            IC = [time_steps[-1].Agi, time_steps[-1].C]
        eval_points = np.linspace(hr, hr + 1, 1000)
        t_span = (hr, hr + 1)
        sol = solve_ivp(lambda t, y: d_dt_system(t, y, ka, kd, rate_elim), t_span, IC, dense_output=True, method='RK45')
        z = sol.sol(eval_points)
        for i in range(len(eval_points)):
            time_steps.append(TimeStep(eval_points[i], z[0][i], z[1][i]))
        time_since_dose += 1

    dt = eval_points[1] - eval_points[0]
    gi_conc = np.array([x.Agi for x in time_steps])
    ab_conc = np.array([x.C for x in time_steps])
    t = np.array([x.t for x in time_steps])
    mic_r = np.array([tvmic_mg for x in time_steps])
    p_time_above = percent_t_above_mic(ab_conc, weight, dt, sim_time)
    print(p_time_above)
    plt.plot(t, ab_conc, label='AMOX in compartment')
    plt.plot(t, gi_conc, label='AMOX in gi tract')
    plt.plot(t, mic_r, label='MIC')
    plt.legend()
    plt.show()
    return p_time_above


def main():
    # NONMEM_Based(tvka, tvweight, tvCl, 1)
    num_subjects = 100
    for i in range(num_subjects):
        eta = [np.random.normal(0, i) for i in omega]
        ka = tvka * np.exp(eta[0])
        age = tvage * np.exp(eta[5])
        weight = tvweight * np.exp(eta[4])
        Cl = ((140 - age)*weight) * theta2 / (72 * 102) * np.exp(eta[1])
        Vd = weight * theta3
        rate_elim = Cl / Vd
        print(f'age: {age}, weight: {weight}, Cl: {Cl}, Vd: {Vd}, rate_elim: {rate_elim}')
        NONMEM_Based(ka, tvkd * np.exp(eta[3]), weight, rate_elim)


if __name__ == '__main__':
    main()
