import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

def run_simulation_normal(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

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
        # 打印每个组成部分的结果以及它们的.value
        for i, t in enumerate(total_time):
            print(f"Component {i}: {t.value}")

        optimized_total_delay = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: normal, 计算的总时延: {optimized_total_delay}")
        print(f"Objective value from solver: {m.options.objfcnval}")
        return optimized_total_delay, 0, 0, 0
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

def run_simulation_local(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

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

    total_time = [ T_i_L[i]  for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: local, 计算的总时延: {total_time_value}")
        return total_time_value, total_time_value, 0, 0
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

def run_simulation_rsu(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

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

    total_time = [ T_i_R[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: rsu, 计算的总时延: {total_time_value}")
        return total_time_value, 0, total_time_value, 0
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

def run_simulation_haps(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

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

    total_time = [ T_i_H[i] for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        total_time_value = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: haps, 计算的总时延: {total_time_value}")
        return total_time_value, 0, 0, total_time_value
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

def run_simulation_pinjun(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

    #y_iL = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    #y_iR = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    #y_iH = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
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
        # 打印每个组成部分的结果以及它们的.value
        for i, t in enumerate(total_time):
            print(f"Component {i}: {t.value}")

        optimized_total_delay = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: pinjun, 计算的总时延: {optimized_total_delay}")
        print(f"Objective value from solver: {m.options.objfcnval}")
        return optimized_total_delay, 0, 0, 0
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

def run_simulation_wuRSU(I, cash1, cash2, cash3, epsilon_values, Omega_values):
    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    epsilon_i_t = [m.Param(value=epsilon_values[i]) for i in range(I)]
    Omega_i_t = [m.Param(value=Omega_values[i]) for i in range(I)]

    R_1 = R_5 = 2000
    R_2 = R_3 = 5000
    R_4 = R_6 = 1000
    F_cav = 5000
    F_rsu = 8000
    F_haps = 10000

    y_iL = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
    #y_iR = [m.Var(value=0, lb=0, ub=1) for i in range(I)]
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



    T_i_H = [(1 - a_i[i])  * epsilon_i_t[i] / (R_3 * b_5[i]) +
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
    #m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    for i in range(I):
        m.Equation(y_iL[i] + y_iH[i]  == 1)

    m.solve(disp=True)

    if m.options.APPSTATUS == 1:
        # 打印每个组成部分的结果以及它们的.value
        for i, t in enumerate(total_time):
            print(f"Component {i}: {t.value}")

        optimized_total_delay = m.options.objfcnval

        print(f"CAV数量: {I}, 模式: wuRSU, 计算的总时延: {optimized_total_delay}")
        print(f"Objective value from solver: {m.options.objfcnval}")
        return optimized_total_delay, 0, 0, 0
    else:
        print("求解失败，请检查模型定义或参数设置")
        return None, 0, 0, 0

# 设置不同的车辆数量进行仿真
vehicle_counts = [1,5,10,15,20,25]

# 存储不同模式下的总时延
delays_normal = []
delays_local = []
delays_rsu = []
delays_haps = []
delays_pinjun = []
delays_wuRSU = []

for I in vehicle_counts:
    # 动态生成线性增加的 epsilon_values 和 Omega_values
    epsilon_values = np.linspace(1000, 1000 + 100 * (I - 1), I)
    Omega_values = np.linspace(1000, 1000 + 100 * (I - 1), I)

    # 正常模式
    total_delay, _, _, _ = run_simulation_normal(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_normal.append(total_delay)
    # 全部本地执行
    total_delay, _, _, _ = run_simulation_local(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_local.append(total_delay)
    # 全部在RSU执行
    total_delay, _, _, _ = run_simulation_rsu(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_rsu.append(total_delay)
    # 全部在HAPS执行
    total_delay, _, _, _ = run_simulation_haps(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_haps.append(total_delay)
    # 三者平均执行
    total_delay, _, _, _ = run_simulation_pinjun(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_pinjun.append(total_delay)
    # 没有rsu
    total_delay, _, _, _ = run_simulation_wuRSU(I, 60000, 60000, 60000, epsilon_values, Omega_values)
    delays_wuRSU.append(total_delay)

# 绘制不同模式下的总时延变化图
plt.figure(figsize=(12, 8))
plt.plot(vehicle_counts, delays_normal, label='Proposed', marker='o', linewidth=2)  # 设置线条宽度为2
plt.plot(vehicle_counts, delays_local, label='All Local', marker='s', linewidth=2)
plt.plot(vehicle_counts, delays_rsu, label='All RSU', marker='^', linewidth=2)
plt.plot(vehicle_counts, delays_haps, label='All HAPS', marker='x', linewidth=2)
plt.plot(vehicle_counts, delays_pinjun, label='AVERAGE', marker='*', linewidth=2)
plt.plot(vehicle_counts, delays_wuRSU, label='wuRSU', marker='+', linewidth=2)
# 设置X和Y轴的名称和单位
plt.xlabel('Number of Vehicles ', fontsize=25, fontname='Times New Roman')
plt.ylabel('Total Delay (ms)', fontsize=25, fontname='Times New Roman')

# 设置坐标轴刻度和标签的字体大小
plt.xticks(vehicle_counts, fontsize=25, fontname='Times New Roman')
plt.yticks(fontsize=25, fontname='Times New Roman')

# 显示图例
plt.legend(fontsize=50, prop={'family': 'Times New Roman'})

# 显示网格
plt.grid(True)

# 设置坐标轴范围
plt.xlim(5, max(vehicle_counts))  # 设置横坐标起始值
plt.ylim(0, 200)  # 设置纵轴范围从 0 到 200


# 设置横坐标刻度为整数
plt.xticks(vehicle_counts)

# 保存图像为EPS格式
plt.savefig('total_delay_vs_vehicles.eps', format='eps')

# 显示图像
plt.show()