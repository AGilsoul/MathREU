import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# NonMEM parameters
tvage = 45
tvka = 0.635  # first order absorption rate constant
tvkd = 1.300
tvVd = 27.7  # drug's volume of distribution in L
tvCl = 21.3  # drug clearance in L/hr
tvrate_elim = 0.27  # Cl/Vd
mic = 1  # mg/mL
tvweight = 70  # weight in kg
tvmic_mg = mic * tvweight  # mic in mg

# NonMEM parameters
omega = [0.04, 0.036, 0.01, 0.04, 0.001, 0.025]
theta1 = 0.635  # KA (1/hr)
theta2 = 15.5   # CL (1/hr)
theta3 = 0.33   # V2


# TimeStep object for storing relevant data at each timestep
# Agi is concentration in GI tract, C is concentration in main compartment
class TimeStep:
    def __init__(self, t, Agi, C):
        self.t = t
        self.C = C
        self.Agi = Agi


# dC/dt, rate of drug absorbed into the body based on amount of drug in the GI tract
# elimination rate is also denoted K for the compartment
def d_dt_system(t, IC, ka, rate_elim):
    Agi = IC[0]
    C = IC[1]
    return [-ka*Agi, ka*Agi - rate_elim * C]


# converts a dose (amount?) to a dosage (concentration?)
def dose_to_dosage(amt, weight):
    return amt / weight


# calculates time above MIC
def percent_t_above_mic(conc, weight, dt, total_time):
    dosage = dose_to_dosage(conc, weight)
    time_above = sum([1 * dt if x > mic else 0 for x in dosage])
    return (time_above / total_time) * 100


# NonMEM based simulation, takes in absorption rate, weight, and elimination rate
# ASSUME CLCR IS SAME AS CL
def NONMEM_Based(ka, weight, rate_elim):
    sim_time = 240    # total simulation time in hours
    dose_amt = 1000  # dose amount in milligrams
    dose_interval = 8  # number of hours between doses
    time_steps = []  # list to store all time steps
    time_since_dose = 0  # stores time since last dose
    delay = 0  # current number of hours to delay
    # for every hour in the simulation
    for hr in range(sim_time):
        # if on the fourth day, miss a dose
        if hr == 96:
            delay = 8
        # if the first hour, set up initial conditions accordingly
        if hr == 0:
            IC = [dose_amt, 0]
            time_since_dose = 0
        # if time to take a dose, set up initial conditions as last values plus a dose
        elif time_since_dose % (dose_interval + delay) == 0:
            IC = [time_steps[-1].Agi + dose_amt, time_steps[-1].C]
            time_since_dose = 0
            delay = 0
        # if not time to set up a dose, set up initial conditions as last values
        else:
            IC = [time_steps[-1].Agi, time_steps[-1].C]
        # set up points to evaluate at
        eval_points = np.linspace(hr, hr + 1, 1000)
        # solve the system of equations for the current hour
        sol = solve_ivp(lambda t, y: d_dt_system(t, y, ka, rate_elim), (hr, hr + 1), IC, dense_output=True, method='RK45')
        # get the solution at the points
        z = sol.sol(eval_points)
        # create TimeStep objects for each point, add to list
        for i in range(len(eval_points)):
            time_steps.append(TimeStep(eval_points[i], z[0][i], z[1][i]))
        # increment time since dose
        time_since_dose += 1

    # get dt for calculation purposes
    dt = eval_points[1] - eval_points[0]
    # get concentration of AMOX in GI tract
    gi_conc = np.array([x.Agi for x in time_steps])
    # get concentration of AMOX in compartment
    ab_conc = np.array([x.C for x in time_steps])
    # get timesteps
    t = np.array([x.t for x in time_steps])
    # get resistant MIC values for each timestep
    mic_r = np.array([tvmic_mg for _ in time_steps])
    # calculate percent of time above MIC
    p_time_above = percent_t_above_mic(ab_conc, weight, dt, sim_time)
    print(p_time_above)
    # plot all values over simulation time
    plt.plot(t, ab_conc, label='AMOX in compartment')
    plt.plot(t, gi_conc, label='AMOX in gi tract')
    plt.plot(t, mic_r, label='MIC')
    plt.legend()
    plt.show()
    return p_time_above


def main():
    # number of subjects to simulate
    num_subjects = 100
    # for every subject
    for i in range(num_subjects):
        # get normally distributed eta parameters
        eta = [np.random.normal(0, i) for i in omega]
        # get patient absorption rate
        ka = tvka * np.exp(eta[0])
        # get patient age
        age = tvage * np.exp(eta[5])
        # get patient weight
        weight = tvweight * np.exp(eta[4])
        # get patient clearance
        Cl = ((140 - age)*weight) * theta2 / (72 * 102) * np.exp(eta[1])
        # get patient volume of distribution
        Vd = weight * theta3
        # get patient rate of elimination
        rate_elim = Cl / Vd
        print(f'age: {age}, weight: {weight}, Cl: {Cl}, Vd: {Vd}, rate_elim: {rate_elim}')
        # run simulation for patient
        NONMEM_Based(ka, weight, rate_elim)


if __name__ == '__main__':
    main()
