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

    m.solve(disp=True)  # 确保调用此方法解决模型，并显示求解信息

    time_L_list = []
    time_R_list = []
    time_H_list = []

    adjusted_optimal_value = 0
    for i in range(I):
        time_L = time_R = time_H = 0

        if b_2[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * s_i1[i].value[0] * Omega_values[i] / (R_1 * b_2[i].value[0])
        if b_1[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * (1 - s_i1[i].value[0]) * Omega_values[i] / (R_3 * b_1[i].value[0])
        time_L += (epsilon_values[i] + Omega_values[i]) / F_cav

        if b_3[i].value[0] > 0:
            time_R += (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_1 * b_3[i].value[0])

        if b_3[i].value[0] > 0 and c_i[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * c_i[i].value[0] * Omega_values[i] / (R_1 * b_3[i].value[0])

        if b_4[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * (1 - c_i[i].value[0]) * Omega_values[i] / (R_2 * b_4[i].value[0])

        time_R += (epsilon_values[i] + Omega_values[i]) / F_rsu

        if b_6[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * s_i2[i].value[0] * epsilon_values[i] / (R_2 * b_6[i].value[0])
        if b_5[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_3 * b_5[i].value[0])
        time_H += (epsilon_values[i] + Omega_values[i]) / F_haps

        adjusted_optimal_value +=  time_L

        time_L_list.append(time_L)
        time_R_list.append(time_R)
        time_H_list.append(time_H)

    total_time_L = sum(time_L_list)
    total_time_R = sum(time_R_list)
    total_time_H = sum(time_H_list)
    total_time = total_time_L

    print(
        f"CAV数量: {I}, 模式: loacl, 总时延: {total_time}, 本地时延: {total_time_L}, RSU时延: {0}, HAPS时延: {0}")

    return total_time, total_time_L, total_time_R, total_time_H


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

    total_time = [  T_i_R[i]  for i in range(I)]
    m.Minimize(sum(total_time))

    m.Equation(sum([epsilon_i_t[i] + Omega_i_t[i] * c_i[i] for i in range(I)]) <= cash1)
    m.Equation(sum([epsilon_i_t[i] * s_i2[i] + Omega_i_t[i] * s_i1[i] for i in range(I)]) <= cash2)
    m.Equation(sum([Omega_i_t[i] + a_i[i] * epsilon_i_t[i] for i in range(I)]) <= cash3)

    m.solve(disp=True)  # 确保调用此方法解决模型
    time_L_list = []
    time_R_list = []
    time_H_list = []

    adjusted_optimal_value = 0
    for i in range(I):
        time_L = time_R = time_H = 0

        if b_2[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * s_i1[i].value[0] * Omega_values[i] / (R_1 * b_2[i].value[0])
        if b_1[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * (1 - s_i1[i].value[0]) * Omega_values[i] / (R_3 * b_1[i].value[0])
        time_L += (epsilon_values[i] + Omega_values[i]) / F_cav

        if b_3[i].value[0] > 0:
            time_R += (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_1 * b_3[i].value[0])

        if b_3[i].value[0] > 0 and c_i[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * c_i[i].value[0] * Omega_values[i] / (R_1 * b_3[i].value[0])

        if b_4[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * (1 - c_i[i].value[0]) * Omega_values[i] / (R_2 * b_4[i].value[0])

        time_R += (epsilon_values[i] + Omega_values[i]) / F_rsu

        if b_6[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * s_i2[i].value[0] * epsilon_values[i] / (R_2 * b_6[i].value[0])
        if b_5[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_3 * b_5[i].value[0])
        time_H += (epsilon_values[i] + Omega_values[i]) / F_haps

        adjusted_optimal_value +=   time_R

        time_L_list.append(time_L)
        time_R_list.append(time_R)
        time_H_list.append(time_H)

    total_time_L = sum(time_L_list)
    total_time_R = sum(time_R_list)
    total_time_H = sum(time_H_list)
    total_time =  total_time_R

    print(
        f"CAV数量: {I}, 模式: rsu, 总时延: {total_time}, 本地时延: {0}, RSU时延: {total_time_R}, HAPS时延: {0}")

    return total_time, total_time_L, total_time_R, total_time_H


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

    m.solve(disp=True)  # 确保调用此方法解决模型

    time_L_list = []
    time_R_list = []
    time_H_list = []

    adjusted_optimal_value = 0
    for i in range(I):
        time_L = time_R = time_H = 0

        if b_2[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * s_i1[i].value[0] * Omega_values[i] / (R_1 * b_2[i].value[0])
        if b_1[i].value[0] > 0:
            time_L += (1 - c_i[i].value[0]) * (1 - s_i1[i].value[0]) * Omega_values[i] / (R_3 * b_1[i].value[0])
        time_L += (epsilon_values[i] + Omega_values[i]) / F_cav

        if b_3[i].value[0] > 0:
            time_R += (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_1 * b_3[i].value[0])

        if b_3[i].value[0] > 0 and c_i[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * c_i[i].value[0] * Omega_values[i] / (R_1 * b_3[i].value[0])

        if b_4[i].value[0] > 0:
            time_R += (1 - s_i1[i].value[0]) * (1 - c_i[i].value[0]) * Omega_values[i] / (R_2 * b_4[i].value[0])

        time_R += (epsilon_values[i] + Omega_values[i]) / F_rsu

        if b_6[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * s_i2[i].value[0] * epsilon_values[i] / (R_2 * b_6[i].value[0])
        if b_5[i].value[0] > 0:
            time_H += (1 - a_i[i].value[0]) * (1 - s_i2[i].value[0]) * epsilon_values[i] / (R_3 * b_5[i].value[0])
        time_H += (epsilon_values[i] + Omega_values[i]) / F_haps

        adjusted_optimal_value +=  time_H

        time_L_list.append(time_L)
        time_R_list.append(time_R)
        time_H_list.append(time_H)

    total_time_L = sum(time_L_list)
    total_time_R = sum(time_R_list)
    total_time_H = sum(time_H_list)
    total_time =  total_time_H

    print(
        f"CAV数量: {I}, 模式: haps， 总时延: {total_time}, 本地时延: {0}, RSU时延: {0}, HAPS时延: {total_time_H}")

    return total_time, total_time_L, total_time_R, total_time_H





# 设置不同的车辆数量进行仿真
vehicle_counts = [1,2,3,4,5,6,7,8,9,10]

# 特殊值进行验算
fixed_epsilon_value = 1000
fixed_Omega_value = 1000
epsilon_values = [fixed_epsilon_value] * 10
Omega_values = [fixed_Omega_value] * 10

# 存储不同模式下的总时延
delays_normal = []
delays_local = []
delays_rsu = []
delays_haps = []

for I in vehicle_counts:
    # 正常模式
    total_delay, _, _, _ = run_simulation_normal(I, 60000, 60000, 60000, epsilon_values[:I], Omega_values[:I])
    delays_normal.append(total_delay)
    # 全部本地执行
    total_delay, _, _, _ = run_simulation_local(I, 60000, 60000, 60000, epsilon_values[:I], Omega_values[:I])
    delays_local.append(total_delay)
    # 全部在RSU执行
    total_delay, _, _, _ = run_simulation_rsu(I, 60000, 60000, 60000, epsilon_values[:I], Omega_values[:I])
    delays_rsu.append(total_delay)
    # 全部在HAPS执行
    total_delay, _, _, _ = run_simulation_haps(I, 60000, 60000, 60000, epsilon_values[:I], Omega_values[:I])
    delays_haps.append(total_delay)

# 绘制不同模式下的总时延变化图
plt.figure(figsize=(12, 8))
plt.plot(vehicle_counts, delays_normal, label='Normal', marker='o')
plt.plot(vehicle_counts, delays_local, label='All Local', marker='s')
plt.plot(vehicle_counts, delays_rsu, label='All RSU', marker='^')
plt.plot(vehicle_counts, delays_haps, label='All HAPS', marker='x')
plt.xlabel('Number of Vehicles (I)')
plt.ylabel('Total Delay')
plt.title('Total Delay vs. Number of Vehicles under Different Modes')
plt.legend()
plt.grid(True)
plt.show()
