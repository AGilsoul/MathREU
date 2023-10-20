import numpy as np

k = 5
days = 10
num_hours = 24 * days  # 7 days in hours
test_range = np.linspace(0, num_hours, 10**k)
set_missed_doses = [2]
q = 0.1
p = 1-q


class PD:
    b_0 = 1
    k = 4
    K = 1
    r = 0.1155
    d = 0.0875
    zMIC = 0.7


class PK:
    tau = 8
    D = 1
    k_e = 0.48 / tau

