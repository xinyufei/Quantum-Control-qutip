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

n = 6
num_edges = 3
seed = 1

if seed == 0:
    Jij, edges = generate_Jij_MC(n, num_edges, 100)
    print(Jij)
else:
    Jij = generate_Jij(n, seed)
    # C = get_ham(n, True, Jij)
    # B = get_ham(n, False, Jij)
diag = get_diag(Jij)
print("minimum energy", min(diag))
# exit()

C = get_ham(n, True, Jij)
B = get_ham(n, False, Jij)
y = uniform(n)
real_init = np.array(y[0:2 ** n])
imag_init = np.array(y[2 ** n:])

tf = 5
T = 100
N = 2
M = 2 ** n + 1

tl = 3600

RealB = B.real
ImagB = B.imag
Brows, Bcols = np.nonzero(RealB)
print("#nonzero REAL elements of B")
for ii in range(len(Brows)):
    print("let RealB[", Brows[ii] + 1, ",", Bcols[ii] + 1, "] := ", RealB[Brows[ii], Bcols[ii]], ";")
Brows, Bcols = np.nonzero(ImagB)
print("#nonzero IMAGINARY elements of B")
for ii in range(len(Brows)):
    print("let ImagB[", Brows[ii] + 1, ",", Bcols[ii] + 1, "] := ", ImagB[Brows[ii], Bcols[ii]], ";")
RealC = C.real
ImagC = C.imag
Crows, Ccols = np.nonzero(RealC)
print("#nonzero REAL elements of C")
for ii in range(len(Crows)):
    print("let RealC[", Crows[ii] + 1, ",", Ccols[ii] + 1, "] := ", RealC[Crows[ii], Ccols[ii]], ";")
Crows, Ccols = np.nonzero(ImagC)
print("#nonzero IMAGINARY elements of C")
for ii in range(len(Crows)):
    print("let ImagC[", Crows[ii] + 1, ",", Ccols[ii] + 1, "] := ", ImagC[Crows[ii], Ccols[ii]], ";")

matrix_exp = []
s_b, v_b = np.linalg.eigh(B)
matrix_exp.append(np.dot(v_b.dot(np.diag(np.exp(-1j * s_b * tf / T))), v_b.conj().T))
s_c, v_c = np.linalg.eigh(C)
matrix_exp.append(np.dot(v_c.dot(np.diag(np.exp(-1j * s_c * tf / T))), v_c.conj().T))
# print(matrix_exp)
RealB = matrix_exp[0].real
ImagB = matrix_exp[0].imag
Brows, Bcols = np.nonzero(RealB)
print("#nonzero REAL elements of exp(B)")
for ii in range(len(Brows)):
    print("let RealexpB[", Brows[ii] + 1, ",", Bcols[ii] + 1, "] := ", RealB[Brows[ii], Bcols[ii]], ";")
Brows, Bcols = np.nonzero(ImagB)
print("#nonzero IMAGINARY elements of exp(B)")
for ii in range(len(Brows)):
    print("let ImagexpB[", Brows[ii] + 1, ",", Bcols[ii] + 1, "] := ", ImagB[Brows[ii], Bcols[ii]], ";")
RealC = matrix_exp[1].real
ImagC = matrix_exp[1].imag
Crows, Ccols = np.nonzero(RealC)
print("#nonzero REAL elements of exp(C)")
for ii in range(len(Crows)):
    print("let RealexpC[", Crows[ii] + 1, ",", Ccols[ii] + 1, "] := ", RealC[Crows[ii], Ccols[ii]], ";")
Crows, Ccols = np.nonzero(ImagC)
print("#nonzero IMAGINARY elements of exp(C)")
for ii in range(len(Crows)):
    print("let ImagexpC[", Crows[ii] + 1, ",", Ccols[ii] + 1, "] := ", ImagC[Crows[ii], Ccols[ii]], ";")
# u = np.zeros((2, T))
# u[0, 0] = 0.000846066
# u[0, 1] = 4.22218e-06
# u[0, 2] = 0.402107
# u[0, 3] = 0.376782
# u[0, 4] = 0.359316
# u[0, 5] = 2.1323e-05
# u[0, 6] = 0.35433
# u[0, 7] = 3.0014e-05
# u[0, 8] = 3.44435e-05
# u[0, 9] = 3.90294e-05
# u[0, 10] = 0.37092
# u[0, 11] = 4.82998e-05
# u[0, 12] = 0.383501
# u[0, 13] = 0.39523
# u[0, 14] = 6.05073e-05
# u[0, 15] = 6.40707e-05
# u[0, 16] = 6.79302e-05
# u[0, 17] = 0.446743
# u[0, 18] = 0.470399
# u[0, 19] = 7.58106e-05
# u[0, 20] = 0.51448
# u[0, 21] = 7.58397e-05
# u[0, 22] = 7.49731e-05
# u[0, 23] = 7.44579e-05
# u[0, 24] = 7.429e-05
# u[0, 25] = 7.44669e-05
# u[0, 26] = 7.49877e-05
# u[0, 27] = 0.645303
# u[0, 28] = 0.678601
# u[0, 29] = 7.00543e-05
# u[0, 30] = 0.736226
# u[0, 31] = 5.94967e-05
# u[0, 32] = 5.31472e-05
# u[0, 33] = 4.70546e-05
# u[0, 34] = 4.11903e-05
# u[0, 35] = 0.865268
# u[0, 37] = 2.81962e-05
# u[0, 36] = 2.10108e-05
# u[0, 38] = 1.39377e-05
# u[0, 39] = 6.94462e-06
# for t in range(T):
#     u[1, t] = 1 - u[0, t]
# u0 = np.loadtxt("../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_SUR.csv", delimiter=',').T
# for t in range(T):
#     print("let u0[", t, "] := ", int(u0[0, t]), ";")
# X = [y[0:2 ** n]]
# for t in range(T):
#     X_t = sum(u[j, t] * matrix_exp[j] for j in range(N)).dot(X[t])
#     print(matrix_exp[1].dot(X[0]))
#     X.append(X_t)
#     print(X[0], X[1])
# bin_result = time_evolution(np.zeros((2**n, 2**n), dtype=complex), [B, C], T, tf, u.T, y[0:2 ** n], False, 1)
# print(compute_obj_energy(C, bin_result))
# print(compute_obj_energy(C, X[-1]))
exit()

# print(matrix_exp[1], expm(-1j * C * tf / T))
# exp = [expm(-1j * B * tf / T), expm(-1j * C * tf / T)]

energy = gp.Model()
# define binary variables
u = energy.addVars(N, T, vtype=gp.GRB.BINARY)
for t in range(T):
    u[0, t].Start = u0[0, t]
    u[1, t].Start = u0[1, t]
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
# print(real_op, imag_op)
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
        energy.addConstrs(reform_real_op_list[row][col] + u[j, k] * M >= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(reform_real_op_list[row][col] - u[j, k] * M <= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(real_op_list[row][col] - reform_real_op_list[row][col] + (1 - u[j, k]) * M >= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(real_op_list[row][col] - reform_real_op_list[row][col] - (1 - u[j, k]) * M <= 0
                          for row in range(2 ** n) for col in range(2 ** n))

        energy.addConstrs(reform_imag_op_list[row][col] + u[j, k] * M >= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(reform_imag_op_list[row][col] - u[j, k] * M <= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(imag_op_list[row][col] - reform_imag_op_list[row][col] + (1 - u[j, k]) * M >= 0
                          for row in range(2 ** n) for col in range(2 ** n))
        energy.addConstrs(imag_op_list[row][col] - reform_imag_op_list[row][col] - (1 - u[j, k]) * M <= 0
                          for row in range(2 ** n) for col in range(2 ** n))
# energy.addConstrs(reform_real_op[k][j].tolist()[row][col] + u[j, k] * M >= 0
#                   for j in range(N) for k in range(T) for row in range(2 ** n) for col in range(2 ** n))
# energy.addConstrs(reform_real_op[k][j][:, col] - u[j, k] * M <= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
# energy.addConstrs(real_op[k][:, col] - reform_real_op[k][j][:, col] + (1 - u[j, k]) * M >= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
# energy.addConstrs(real_op[k][:, col] - reform_real_op[k][j][:, col] - (1 - u[j, k]) * M <= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
#
# energy.addConstrs(reform_imag_op[k][j][:, col] + u[j, k] * M >= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
# energy.addConstrs(reform_imag_op[k][j][:, col] - u[j, k] * M <= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
# energy.addConstrs(imag_op[k][:, col] - reform_imag_op[k][j][:, col] + (1 - u[j, k]) * M >= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))
# energy.addConstrs(imag_op[k][:, col] - reform_imag_op[k][j][:, col] - (1 - u[j, k]) * M <= 0
#                   for j in range(N) for k in range(T) for col in range(2 ** n))

# state constraints
energy.addConstrs(real_op[k + 1][:, col] - sum(
    matrix_exp[j].real @ reform_real_op[k][j][:, col] - matrix_exp[j].imag @ reform_imag_op[k][j][:, col] 
    for j in range(N)) == 0 for k in range(T) for col in range(2 ** n))
energy.addConstrs(imag_op[k + 1][:, col] - sum(
    matrix_exp[j].real @ reform_imag_op[k][j][:, col] + matrix_exp[j].imag @ reform_real_op[k][j][:, col] 
    for j in range(N)) == 0 for k in range(T) for col in range(2 ** n))

# sos1 constraints
energy.addConstrs(sum(u[j, k] for j in range(N)) == 1 for k in range(T))
energy.addConstr(final_real_state @ C @ final_real_state + final_imag_state @ C @ final_imag_state >= min(diag))
energy.addConstr(final_real_state @ C @ final_real_state + final_imag_state @ C @ final_imag_state <= 0)
energy.Params.NonConvex = 2
energy.Params.TimeLimit = tl
energy.Params.Crossover = 0
energy.Params.LogFile = "../output/gurobi/Energyreformtest{}_evotime{}_n_ts{}_instance{}_tl{}.log".format(n, tf, T, seed, tl)
energy.optimize()

u_val = np.zeros((T, N))
for j in range(N):
    for k in range(T):
        u_val[k, j] = u[j, k].x
np.savetxt("../output/gurobi/Energyreformtest{}_evotime{}_n_ts{}_instance{}_tl{}.csv".format(n, tf, T, seed, tl), u_val,
           delimiter=',')
# bin_result = np.zeros(2 ** n, dtype=complex)
# for j in range(N):
#     for k in range(1, T + 1):
#         print(real_op[k].getAttr('X') + 1j * imag_op[k].getAttr('X'))
# print("================================")
# bin_result = time_evolution(np.zeros((2**n, 2**n), dtype=complex), [B, C], T, tf, u_val, y[0:2 ** n], False, 1)
# print(compute_obj_energy(C, bin_result))
