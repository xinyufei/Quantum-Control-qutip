import argparse
import os
import sys
import time
import gurobipy as gp
import numpy as np

sys.path.append("../..")
from tools.auxiliary_energy import *
from scipy.linalg import expm
from tools.evolution import compute_TV_norm, compute_obj_by_switch, time_evolution, compute_obj_energy

n = 2
num_edges = 1

Jij, edges = generate_Jij_MC(n, num_edges, 100)
print(Jij)

C = get_ham(n, True, Jij)
B = get_ham(n, False, Jij)
y = uniform(n)
real_init = np.array(y[0:2 ** n])
imag_init = np.array(y[2 ** n:])

tf = 2
T = 40
N = 2
M = 2 ** n + 1

matrix_exp = []
s_b, v_b = np.linalg.eigh(B)
matrix_exp.append(np.dot(v_b.dot(np.diag(np.exp(-1j * s_b * tf / T))), v_b.conj().T))
s_c, v_c = np.linalg.eigh(C)
matrix_exp.append(np.dot(v_c.dot(np.diag(np.exp(-1j * s_c * tf / T))), v_c.conj().T))
# print(matrix_exp[1], expm(-1j * C * tf / T))
# exp = [expm(-1j * B * tf / T), expm(-1j * C * tf / T)]

energy = gp.Model()
# define binary variables
u = energy.addVars(N, T, vtype=gp.GRB.BINARY)
# define real part and imaginary part of operator
real_op, imag_op, reform_real_op, reform_imag_op = [np.eye(2 ** n)], [np.zeros((2 ** n, 2 ** n))], [], []
for k in range(T):
    real_op.append(energy.addMVar((2 ** n, 2 ** n), lb=-np.Infinity, ub=np.Infinity))
    imag_op.append(energy.addMVar((2 ** n, 2 ** n), lb=-np.Infinity, ub=np.Infinity))
    reform_real_op.append([])
    reform_imag_op.append([])
    for j in range(N):
        reform_real_op[k].append(energy.addMVar((2 ** n, 2 ** n), lb=-np.Infinity, ub=np.Infinity))
        reform_imag_op[k].append(energy.addMVar((2 ** n, 2 ** n), lb=-np.Infinity, ub=np.Infinity))
print(real_op, imag_op)
final_real_state = energy.addMVar(2 ** n, lb=-np.Infinity, ub=np.Infinity)
final_imag_state = energy.addMVar(2 ** n, lb=-np.Infinity, ub=np.Infinity)

energy.setObjective(final_real_state @ C @ final_real_state + final_imag_state @ C @ final_imag_state)

# final state constraints
energy.addConstrs(real_op[-1][row, :] @ real_init - final_real_state[row] == 0 for row in range(2 ** n))
energy.addConstrs(imag_op[-1][row, :] @ real_init - final_imag_state[row] == 0 for row in range(2 ** n))

# reformulated constraints
# print(reform_real_op[0][0].tolist())
# print(reform_real_op[0][0][:, 0], np.ones(2 ** n) * u[0, 0].tolist())
for j in range(N):
    for k in range(T):
        reform_real_op_list = reform_real_op[k][j].tolist()
        reform_imag_op_list = reform_imag_op[k][j].tolist()
        real_op_list = real_op[k].tolist()
        imag_op_list = imag_op[k].tolist()
        energy.addConstr(sum((reform_imag_op_list[row][col] - imag_op_list[row][col]) ** 2
                             for row in range(2 ** n) for col in range(2 ** n)) <= ((1 - u[j, k]) * M) ** 2)
        energy.addConstr(sum((reform_imag_op_list[row][col] - imag_op_list[row][col]) ** 2
                             for row in range(2 ** n) for col in range(2 ** n)) <= (u[j, k] * M) ** 2)
        energy.addConstr(sum((reform_real_op_list[row][col] - real_op_list[row][col]) ** 2
                             for row in range(2 ** n) for col in range(2 ** n)) <= ((1 - u[j, k]) * M) ** 2)
        energy.addConstr(sum((reform_real_op_list[row][col] - real_op_list[row][col]) ** 2
                             for row in range(2 ** n) for col in range(2 ** n)) <= (u[j, k] * M) ** 2)

# state constraints
energy.addConstrs(real_op[k + 1][:, col] - sum(
    matrix_exp[j].real @ reform_real_op[k][j][:, col] - matrix_exp[j].imag @ reform_imag_op[k][j][:, col] 
    for j in range(N)) == 0 for k in range(T) for col in range(2 ** n))
energy.addConstrs(imag_op[k + 1][:, col] - sum(
    matrix_exp[j].real @ reform_imag_op[k][j][:, col] + matrix_exp[j].imag @ reform_real_op[k][j][:, col] 
    for j in range(N)) == 0 for k in range(T) for col in range(2 ** n))

# sos1 constraints
energy.addConstrs(sum(u[j, k] for j in range(N)) == 1 for k in range(T))
energy.Params.NonConvex = 2
energy.Params.TimeLimit = 3600
energy.Params.LogFile = "../output/gurobi/EnergySOCP2_evotime2.0_n_ts40_instance0_tl3600.log"
energy.optimize()

u_val = np.zeros((T, N))
for j in range(N):
    for k in range(T):
        u_val[k, j] = u[j, k].x
np.savetxt("../output/gurobi/EnergySOCP2_evotime2.0_n_ts40_instance0_tl3600.csv", u_val, delimiter=',')
# bin_result = np.zeros(2 ** n, dtype=complex)
# for j in range(N):
#     for k in range(1, T + 1):
#         print(real_op[k].getAttr('X') + 1j * imag_op[k].getAttr('X'))
# print("================================")
# bin_result = time_evolution(np.zeros((2**n, 2**n), dtype=complex), [B, C], T, tf, u_val, y[0:2 ** n], False, 1)
# print(compute_obj_energy(C, bin_result))
