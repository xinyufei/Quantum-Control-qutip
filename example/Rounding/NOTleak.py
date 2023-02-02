import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from qutip import identity, sigmax, sigmaz, sigmay, tensor, Qobj
from qutip.qip.operations.gates import cnot
sys.path.append("../..")
from tools import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='NOTleak')
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=10)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=200)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# if sos1 property holds
parser.add_argument('--sos1', help='sos1 property holds or not', type=int, default=0)
# rounding type
parser.add_argument('--type', help='type of rounding (SUR, minup, maxswitch)', type=str, default='SUR')
# minimum up time steps
# parser.add_argument('--min_up', help='minimum up time steps', nargs='+', type=int, default=10)
parser.add_argument('--min_up', help='minimum up time steps', type=int, default=10)
# maximum number of switches
parser.add_argument('--max_switch', help='maximum number of switches', type=int, default=10)
# time limit for rounding by Gurobi
parser.add_argument('--time_limit', help='time limit for rounding by Gurobi', type=int, default=60)
args = parser.parse_args()
# args.type = "minup"
# args.initial_control = "../control/Continuous/CNOT_evotime10.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT.csv"
# args.sos1 = 1
# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate with summation one constraint
# The control Hamiltonians (Qobj classes)
# Drift Hamiltonian
H_d_origin = Qobj(
    0 * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) + 2 * math.pi * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))
H_c_origin = [Qobj(1 / 2 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
                   + np.sqrt(2) / 2 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])),
              Qobj(1 / 2 * np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
                   + np.sqrt(2) / 2 * np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]))]
# print(H_d, H_c)
# H_d = H_d_origin - H_c_origin[0] - H_c_origin[1]
# H_c = [2 * hc for hc in H_c_origin]
H_d = H_d_origin
H_c = H_c_origin

# start point for the gate evolution
X_0 = identity(3)
# Target for the gate evolution
X_targ = Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))

# objective value type
obj_type = "UNIT"

if not os.path.exists("../output/Rounding/"):
    os.makedirs("../output/Rounding/")
if not os.path.exists("../control/Rounding/"):
    os.makedirs("../control/Rounding/")
if not os.path.exists("../figure/Rounding/"):
    os.makedirs("../figure/Rounding/")

output_fig = "../figure/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0]
if args.type == "SUR":
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] + "_" + str(args.sos1) + "_SUR.log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] + "_" + str(args.sos1) + "_SUR.csv"
if args.type == "minup":
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                 + "_minup" + str(args.min_up) + "_" + str(args.sos1) + ".log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                     + "_minup" + str(args.min_up) + "_" + str(args.sos1) + ".csv"
if args.type == "maxswitch":
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                 + "_maxswitch" + str(args.max_switch) + "_" + str(args.sos1) + ".log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                     + "_maxswitch" + str(args.max_switch) + "_" + str(args.sos1) + ".csv"

# round the solution
b_rel = np.loadtxt(args.initial_control, delimiter=',')
round = Rounding()
round.build_rounding_optimizer(b_rel, args.evo_time, args.n_ts, args.type, args.min_up, args.max_switch, 
                               time_limit=args.time_limit, out_fig=output_fig, out_num=output_num)
if args.sos1 == 1:
    b_bin, c_time = round.rounding_with_sos1()
else:
    b_bin, c_time = round.rounding_without_sos1()
# b_bin = rounding(b_rel, args.type, args.min_up / args.n_ts, args.max_switch, output_fig=output_fig)

bin_result = time_evolution(H_d.full(), [hc.full() for hc in H_c], args.n_ts, args.evo_time, b_bin, X_0.full(), False, 1)

f = open(output_num, "a+")
print("computational time", c_time, file=f)
print("original objective", compute_obj_fid(X_targ, bin_result, phase="PSU"), file=f)
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()

np.savetxt(output_control, b_bin, delimiter=',')


