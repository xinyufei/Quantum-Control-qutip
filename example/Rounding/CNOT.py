import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
sys.path.append("../..")
from tools import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='CNOTBi')
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

args = parser.parse_args()
# args.type = "minup"
# args.initial_control = "../control/Continuous/CNOT_evotime10.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT.csv"
# args.sos1 = 1
# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate with summation one constraint
# The control Hamiltonians (Qobj classes)
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz()) \
      + tensor(sigmay(), identity(2))
H_c = [tensor(sigmax(), identity(2)) - tensor(sigmay(), identity(2))]
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()

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
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] + "_SUR.log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] + "_SUR.csv"
if args.type == "minup":
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                 + "_minup" + str(args.min_up) + ".log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                     + "_minup" + str(args.min_up) + ".csv"
if args.type == "maxswitch":
    output_num = "../output/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                 + "_maxswitch" + str(args.max_switch) + ".log"
    output_control = "../control/Rounding/" + args.initial_control.split('/')[-1].split('.csv')[0] \
                     + "_maxswitch" + str(args.max_switch) + ".csv"

# round the solution
b_rel = np.loadtxt(args.initial_control, delimiter=',')
round = Rounding()
round.build_rounding_optimizer(b_rel, args.evo_time, args.n_ts, args.type, args.min_up, args.max_switch,
                               out_fig=output_fig)
if args.sos1 == 1:
    b_bin, c_time = round.rounding_with_sos1()
else:
    b_bin, c_time = round.rounding_without_sos1()
# b_bin = rounding(b_rel, args.type, args.min_up / args.n_ts, args.max_switch, output_fig=output_fig)
h_d_mat = (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())).full()
h_c_mat = [tensor(sigmax(), identity(2)).full(), tensor(sigmay(), identity(2)).full()]
bin_result = time_evolution(h_d_mat, h_c_mat, args.n_ts, args.evo_time, b_bin, X_0.full(), False, 1)

f = open(output_num, "w+")
print("computational time", c_time, file=f)
print("original objective", compute_obj_fid(X_targ, bin_result), file=f)
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()

np.savetxt(output_control, b_bin, delimiter=',')


