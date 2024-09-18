import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

# 定义仿真函数
def run_simulation_normal(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    y_iL = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    y_iR = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    y_iH = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * s_i1[i] * Omega_i_t[i] / (R_1 * b_2[i]) +
             (1 - c_i[i]) * (1 - s_i1[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_R = [(1 - s_i2[i]) * epsilon_i_t[i] / (R_1 * b_3[i]) +
             ((1 - s_i1[i]) * (c_i[i] * Omega_i_t[i] / (R_1 * b_3[i]) +
                               (1 - c_i[i]) * Omega_i_t[i] / (R_2 * b_4[i]))) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_rsu for i in range(I)]

    T_i_H = [(1 - a_i[i]) * s_i2[i] * epsilon_i_t[i] / (R_2 * b_6[i]) +
             (1 - a_i[i]) * (1 - s_i2[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [y_iL[i] * T_i_L[i] + y_iR[i] * T_i_R[i] + y_iH[i] * T_i_H[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    for i in range(I):
        m.Equation(y_iL[i] + y_iR[i] + y_iH[i] == 1)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        optimized_total_delay = m.options.objfcnval
        return optimized_total_delay, 0, 0, 0
    else:
        return None, 0, 0, 0

def run_simulation_local(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * s_i1[i] * Omega_i_t[i] / (R_1 * b_2[i]) +
             (1 - c_i[i]) * (1 - s_i1[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_R = [(1 - s_i2[i]) * epsilon_i_t[i] / (R_1 * b_3[i]) +
             ((1 - s_i1[i]) * (c_i[i] * Omega_i_t[i] / (R_1 * b_3[i]) +
                               (1 - c_i[i]) * Omega_i_t[i] / (R_2 * b_4[i]))) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_rsu for i in range(I)]

    T_i_H = [(1 - a_i[i]) * s_i2[i] * epsilon_i_t[i] / (R_2 * b_6[i]) +
             (1 - a_i[i]) * (1 - s_i2[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [T_i_L[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval
        return total_time_value, total_time_value, 0, 0
    else:
        return None, 0, 0, 0

def run_simulation_rsu(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * s_i1[i] * Omega_i_t[i] / (R_1 * b_2[i]) +
             (1 - c_i[i]) * (1 - s_i1[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_R = [(1 - s_i2[i]) * epsilon_i_t[i] / (R_1 * b_3[i]) +
             ((1 - s_i1[i]) * (c_i[i] * Omega_i_t[i] / (R_1 * b_3[i]) +
                               (1 - c_i[i]) * Omega_i_t[i] / (R_2 * b_4[i]))) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_rsu for i in range(I)]

    T_i_H = [(1 - a_i[i]) * s_i2[i] * epsilon_i_t[i] / (R_2 * b_6[i]) +
             (1 - a_i[i]) * (1 - s_i2[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [T_i_R[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval
        return total_time_value, 0, total_time_value, 0
    else:
        return None, 0, 0, 0

def run_simulation_haps(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * s_i1[i] * Omega_i_t[i] / (R_1 * b_2[i]) +
             (1 - c_i[i]) * (1 - s_i1[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_R = [(1 - s_i2[i]) * epsilon_i_t[i] / (R_1 * b_3[i]) +
             ((1 - s_i1[i]) * (c_i[i] * Omega_i_t[i] / (R_1 * b_3[i]) +
                               (1 - c_i[i]) * Omega_i_t[i] / (R_2 * b_4[i]))) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_rsu for i in range(I)]

    T_i_H = [(1 - a_i[i]) * s_i2[i] * epsilon_i_t[i] / (R_2 * b_6[i]) +
             (1 - a_i[i]) * (1 - s_i2[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [T_i_H[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval
        return total_time_value, 0, 0, total_time_value
    else:
        return None, 0, 0, 0

def run_simulation_pinjun(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * s_i1[i] * Omega_i_t[i] / (R_1 * b_2[i]) +
             (1 - c_i[i]) * (1 - s_i1[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_R = [(1 - s_i2[i]) * epsilon_i_t[i] / (R_1 * b_3[i]) +
             ((1 - s_i1[i]) * (c_i[i] * Omega_i_t[i] / (R_1 * b_3[i]) +
                               (1 - c_i[i]) * Omega_i_t[i] / (R_2 * b_4[i]))) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_rsu for i in range(I)]

    T_i_H = [(1 - a_i[i]) * s_i2[i] * epsilon_i_t[i] / (R_2 * b_6[i]) +
             (1 - a_i[i]) * (1 - s_i2[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [0.33 * T_i_L[i] + 0.33 * T_i_R[i] + 0.34 * T_i_H[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        optimized_total_delay = m.options.objfcnval
        return optimized_total_delay, 0, 0, 0
    else:
        return None, 0, 0, 0

def run_simulation_wuRSU(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000

    y_iL = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    y_iH = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    a_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    c_i = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i1 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    s_i2 = [m.Var(value=1, lb=0, ub=1, integer=True) for i in range(I)]
    b_1 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_2 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_3 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_4 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_5 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]
    b_6 = [m.Var(value=0.5, lb=0.0001, ub=1) for i in range(I)]

    T_i_L = [(1 - c_i[i]) * Omega_i_t[i] / (R_3 * b_1[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_cav for i in range(I)]

    T_i_H = [(1 - a_i[i]) * epsilon_i_t[i] / (R_3 * b_5[i]) +
             (epsilon_i_t[i] + Omega_i_t[i]) / F_haps for i in range(I)]

    m.Equation(sum(b_1) <= 1)
    m.Equation(sum(b_2) <= 1)
    m.Equation(sum(b_3) <= 1)
    m.Equation(sum(b_4) <= 1)
    m.Equation(sum(b_5) <= 1)
    m.Equation(sum(b_6) <= 1)

    total_time = [y_iL[i] * T_i_L[i] + y_iH[i] * T_i_H[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    for i in range(I):
        m.Equation(y_iL[i] + y_iH[i] == 1)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        optimized_total_delay = m.options.objfcnval
        return optimized_total_delay, 0, 0, 0
    else:
        return None, 0, 0, 0

# 定义计算能力范围
F_cav_values = np.linspace(3000, 10000, 6)
F_rsu_values = np.linspace(5000, 15000, 6)
F_haps_values = np.linspace(7000, 20000, 6)

# 设置车辆数量为一个中等值，例如 20
I = 20
epsilon_values = np.full(I, 2000)
Omega_values = np.full(I, 2000)
cash1 = cash2 = cash3 = 80000  # 固定缓存值

# 模拟每种计算能力设置
def simulate_and_plot(F_values, fixed_F1, fixed_F2, label_F, filename):
    delays_normal = []
    delays_local = []
    delays_rsu = []
    delays_haps = []
    delays_pinjun = []
    delays_wuRSU = []

    for F in F_values:
        if label_F == 'F_cav':
            F_cav = F
            F_rsu = fixed_F1
            F_haps = fixed_F2
        elif label_F == 'F_rsu':
            F_cav = fixed_F1
            F_rsu = F
            F_haps = fixed_F2
        else:  # F_haps
            F_cav = fixed_F1
            F_rsu = fixed_F2
            F_haps = F

        # 使用相同的计算能力值对所有仿真进行设置
        total_delay, _, _, _ = run_simulation_normal(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_normal.append(total_delay)

        total_delay, _, _, _ = run_simulation_local(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_local.append(total_delay)

        total_delay, _, _, _ = run_simulation_rsu(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_rsu.append(total_delay)

        total_delay, _, _, _ = run_simulation_haps(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_haps.append(total_delay)

        total_delay, _, _, _ = run_simulation_pinjun(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_pinjun.append(total_delay)

        total_delay, _, _, _ = run_simulation_wuRSU(I, cash1, cash2, cash3, epsilon_values, Omega_values, F_cav, F_rsu, F_haps)
        delays_wuRSU.append(total_delay)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(F_values, delays_normal, label='Proposed', marker='o',  linewidth=2)
    plt.plot(F_values, delays_local, label='All Local', marker='s',  linewidth=2)
    plt.plot(F_values, delays_rsu, label='All RSU', marker='^',  linewidth=2)
    plt.plot(F_values, delays_haps, label='All HAPS', marker='x',  linewidth=2)
    plt.plot(F_values, delays_pinjun, label='AVERAGE', marker='*',  linewidth=2)
    plt.plot(F_values, delays_wuRSU, label='wuRSU', marker='+', linewidth=2)
    plt.xlabel(f'{label_F} Value(CPU cycle/s)', fontsize=25, fontname='Times New Roman')
    plt.ylabel('Total Delay(ms)', fontsize=25, fontname='Times New Roman')

    # 显示图例
    plt.legend(fontsize=50, prop={'family': 'Times New Roman'})

    plt.grid(True)
    plt.xlim(min(F_values), max(F_values))  # 设置横坐标范围，使图表紧贴线条

    # 根据不同的图表设置纵坐标范围
    if label_F == 'F_cav':
        plt.ylim(0, 30)  # 第一个图的纵轴设置为0到30
    else:
        plt.ylim(0, 18)  # 剩余两个图的纵轴设置为0到18
    plt.xticks(F_values, fontsize=25, fontname='Times New Roman')  # 设置横坐标刻度
    plt.yticks(fontsize=25, fontname='Times New Roman')

    # 使用tight_layout()来调整图像布局
    plt.tight_layout()

    plt.savefig(f'{filename}.eps', format='eps')  # 高分辨率保存图像
    plt.show()

# 绘制F_cav的影响
simulate_and_plot(F_cav_values, 8000, 10000, 'F_cav', 'F_cav_effect')

# 绘制F_rsu的影响
simulate_and_plot(F_rsu_values, 5000, 10000, 'F_rsu', 'F_rsu_effect')

# 绘制F_haps的影响
simulate_and_plot(F_haps_values, 5000, 8000, 'F_haps', 'F_haps_effect')