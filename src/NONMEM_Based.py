import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


ka = 1.02  # first order absorption rate constant
Vd = 27.7  # drug's volume of distribution in L
Cl = 21.3  # drug clearance in L/hr
rate_elim = 0.27  # Cl/Vd


class TimeStep:
    def __init__(self, t, Agi, C):
        self.t = t
        self.C = C
        self.Agi = Agi


# dC/dt, rate of drug absorbed into the body based on amount of drug in the GI tract
def d_dt_system(t, IC):
    Agi = IC[0]
    C = IC[1]
    return [-ka*Agi, ka*Agi - rate_elim * C]


def NONMEM_Based():
    sim_time = 48    # total simulation time in hours
    dose_amt = 1000  # dose amount in milligrams
    dose_interval = 8  # number of hours between doses
    time_steps = []
    for hr in range(sim_time):
        if hr == 0:
            print(f'hour 0, dose given')
            IC = [dose_amt, 0]
        elif hr % dose_interval == 0:
            print(f'hour {hr}, dose given')
            IC = [time_steps[-1].Agi + dose_amt, time_steps[-1].C]
        else:
            print(f'hour {hr}')
            IC = [time_steps[-1].Agi, time_steps[-1].C]
        eval_points = np.linspace(hr, hr + 1, 1000)
        t_span = (hr, hr + 1)
        sol = solve_ivp(d_dt_system, t_span, IC, dense_output=True, method='RK45')
        z = sol.sol(eval_points)
        for i in range(len(eval_points)):
            time_steps.append(TimeStep(eval_points[i], z[0][i], z[1][i]))

    gi_conc = [x.Agi for x in time_steps]
    ab_conc = [x.C for x in time_steps]
    t = [x.t for x in time_steps]
    plt.plot(t, ab_conc, label='AMOX in compartment')
    plt.plot(t, gi_conc, label='AMOX in gi tract')
    plt.legend()
    plt.show()
    return


def main():
    NONMEM_Based()


if __name__ == '__main__':
    main()
