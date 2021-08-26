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
# rounding type
parser.add_argument('--type', help='type of rounding (SUR, minup, maxswitch)', type=str, default='SUR')
# minimum up time steps
# parser.add_argument('--min_up', help='minimum up time steps', nargs='+', type=int, default=10)
parser.add_argument('--min_up', help='minimum up time steps', type=int, default=10)
# maximum number of switches
parser.add_argument('--max_switch', help='maximum number of switches', type=int, default=10)

args = parser.parse_args()
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

output_fig = "../figure/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0]
if type == "SUR":
    output_num = "../output/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] + "_SUR.log"
    output_control = "../control/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] + "_SUR.log"
if type == "minup":
    output_num = "../output/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] \
                 + "_minup" + str(min_up_times) + ".log"
    output_control = "../control/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] \
                     + "_minup" + str(min_up_times) + ".csv"
if type == "maxswitch":
    output_num = "../output/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] \
                 + "_maxswitch" + str(max_switches) + ".log"
    output_control = "../control/Rounding/" + args.initial_file.split('/')[-1].split('.csv')[0] \
                     + "_maxswitch" + str(max_switches) + ".csv"

# round the solution
b_rel = np.loadtxt(args.initial_control, delimeter=',')
b_bin = rounding(b_rel, args.type, args.min_up / args.n_ts, args.max_switch, output_fig=output_fig)
h_d_mat = (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())).full()
h_c_mat = [tensor(sigmax(), identity(2)).full(), tensor(sigmay(), identity(2)).full()]
bin_result = time_evolution(h_d_mat, h_c_mat, args.n_ts, args.evo_time, b_bin.T, X_0.full(), False, 1)

f = open(output_num, "a+")
print("original objective", compute_obj_fid(X_targ.full(), bin_result), file=f)
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()

np.savetxt(output_control, b_bin.T, delimeter=',')

