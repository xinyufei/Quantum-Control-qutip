import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

sys.path.append("../..")
from switchingtime.switch_time import *
from tools.evolution import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='CNOTST')
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=10)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=200)
# initial type for control variables
parser.add_argument('--initial_type', help='initial type of control variables (rnd, ave, warm)', type=str,
                    default="warm")
# initial control file obtained from ADMM algorithm
parser.add_argument('--admm_control', help='file name of initial control', type=str, default=None)
# minimum up time constraint
parser.add_argument('--min_up_time', help='minimum up time', type=float, default=0)
# tv regularizer parameter
parser.add_argument('--alpha', help='tv regularizer parameter', type=float, default=0.05)

args = parser.parse_args()
# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate with summation one constraint
# The control Hamiltonians (Qobj classes)
# The control Hamiltonians (Qobj classes)
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()

if args.admm_control is None:
    print("Must provide control results of ADMM!")
    exit()

warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time(args.admm_control, args.n_ts / args.evo_time)

# sequence of control hamiltonians
ctrl_hamil = [(H_d + H_c[0]).full(), (H_d + H_c[1]).full()]

if args.initial_type == "ave":
    initial = np.ones(num_switch + 1) * args.evo_time / (num_switch + 1)
if args.initial_type == "rnd":
    initial_pre = np.random.random(num_switch + 1)
    initial = initial_pre.copy() / sum(initial_pre) * args.evo_time
if args.initial_type == "warm":
    initial = warm_start_length

# build optimizer
cnot_opt = SwitchTimeOpt()
cnot_opt.build_optimizer(
    ctrl_hamil, ctrl_hamil_idx, initial, X_0.full(), X_targ.full(), args.evo_time, num_switch, args.min_up_time, None)
start = time.time()
res = cnot_opt.optimize()
end = time.time()

if not os.path.exists("../output/SwitchTime/"):
    os.makedirs("../output/SwitchTime/")
if not os.path.exists("../control/SwitchTime/"):
    os.makedirs("../control/SwitchTime/")
if not os.path.exists("../figure/SwitchTime/"):
    os.makedirs("../figure/SwitchTime/")

# output file
output_name = "../output/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name, str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type, str(args.min_up_time)) + ".log"
output_file = open(output_name, "a+")
print(res, file=output_file)
print("switching time points", cnot_opt.switch_time, file=output_file)
print("computational time", end - start, file=output_file)

# retrieve control
control_name = "../control/SwitchTime/CNOT_evotime" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name, str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type, str(args.min_up_time)) + ".csv"
control = cnot_opt.retrieve_control(args.n_ts)
np.savetxt(control_name, control, delimiter=",")

print("alpha", args.alpha, file=output_file)
tv_norm = cnot_opt.tv_norm()
print("tv norm", tv_norm, file=output_file)
print("objective with tv norm", cnot_opt.obj + args.alpha * tv_norm, file=output_file)

# figure file
figure_name = "../figure/SwitchTime/CNOT_evotime" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name, str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type, str(args.min_up_time)) + ".png"
cnot_opt.draw_control(figure_name)

b_bin = np.loadtxt(control_name, delimiter=",")
f = open(output_name, "a+")
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()