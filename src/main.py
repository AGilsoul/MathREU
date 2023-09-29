import numpy as np
import matplotlib.pyplot as plt


class Model:

    class ParamSet:
        def __init__(self, psi_max, psi_min, k, MIC, zMIC, dose_frequency=0.125, a0_MIC_ratio=5):
            self.psi_max = psi_max
            self.psi_min = psi_min
            self.k = k
            self.MIC = MIC
            self.zMIC = zMIC
            self.a0 = zMIC * a0_MIC_ratio
            self.dose_frequency = dose_frequency
            return


# growth rate of bacterial population
def psi(ab_conc: float, params: Model.ParamSet):
    numerator = (params.psi_max - params.psi_min) * (ab_conc/params.zMIC)**params.k
    denominator = (ab_conc/params.zMIC)**params.k - (params.psi_min/params.psi_max)
    return params.psi_max - (numerator / denominator)


# change in density, X, of a bacterial population
def dX_dt(t: float, X: float, params: Model.ParamSet):
    ab = a(t, params)
    return np.log(10) * psi(ab, params) * X, ab


# antibiotic concentration at time t (hours)
def a(t: float, params: Model.ParamSet):
    time_since_dose = t % (1 / params.dose_frequency)
    if time_since_dose == 0:
        return params.a0
    delta = params.dose_frequency * np.log(0.5*params.MIC/params.a0)
    return params.a0 * np.exp(delta * time_since_dose)


def basic_euler(X_0, params: Model.ParamSet):
    time_range = np.linspace(0, 24, 1_000)
    dt = time_range[1] - time_range[0]
    X = []
    ab = []
    for i in range(0, len(time_range)):
        print(i)
        if i == 0:
            cur_X = X_0
        else:
            cur_X = X[-1]
        dx, a = dX_dt(time_range[i], cur_X, params)
        print(f'dx: {dx}, dt: {dt}')
        X.append(cur_X + dx * dt)
        ab.append(a)

    X = np.array(X)
    ab = np.array(ab)
    print(f'X: {X}')
    print(f'AB: {ab}')

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(time_range, X)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Bacterial Density')
    ax[0].set_xlabel('Time (hrs)')


    base_mic = [params.MIC for _ in time_range]
    base_zmic = [params.zMIC for _ in time_range]
    ax[1].plot(time_range, ab)
    ax[1].plot(time_range, base_mic, label='MIC')
    ax[1].plot(time_range, base_zmic, label='zMIC')
    ax[1].set_ylabel('Antibiotic Concentration')
    ax[1].set_xlabel('Time (hrs)')
    ax[1].legend()
    plt.show()
    return


def main():
    # for Ampicillin
    basic_euler(10**6,
                params=Model.ParamSet(0.75, -4.0, 1, 3.4, 8.0))
    return


if __name__ == '__main__':
    main()