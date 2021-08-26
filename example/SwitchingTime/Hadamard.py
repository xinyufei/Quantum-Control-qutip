import os
import numpy as np
import argparse

import sys

sys.path.append("../..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
# from rounding import rounding
from tools import *
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from switchingtime.switch_time import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='HadamardST')
# number of qubits
parser.add_argument('--qubit_num', help='number of qubits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=8)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=80)
# initial type for control variables 
parser.add_argument('--initial_type', help='initial type of control variables (rnd, ave, warm)', type=str,
                    default="warm")
# initial control file obtained from ADMM algorithm
parser.add_argument('--admm_control', help='file name of initial control', type=str,
                    default="../control/ADMM/SPINWARMNEW2_evotime8_n_ts80_ptypeWARM_offset0_objUNIT_penalty0.001_sum_penalty0.01.csv")
# minimum up time constraint
parser.add_argument('--min_up_time', help='minimum up time', type=float, default=1)
# tv regularizer parameter
parser.add_argument('--alpha', help='tv regularizer parameter', type=float, default=0.05)

args = parser.parse_args()

Hops, H0, U0, U = generate_spin_func(args.qubit_num)

# The control Hamiltonians (Qobj classes)
H_c = [Qobj(hops) for hops in Hops]
# Drift Hamiltonian
H_d = Qobj(H0)
# start point for the gate evolution
X_0 = Qobj(U0)
# Target for the gate evolution
X_targ = Qobj(U)

if args.admm_control is None:
    print("Must provide control results of ADMM!")
    exit()

warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time(args.admm_control,
                                                                      delta_t=args.evo_time / args.n_ts)
print(num_switch)

# sequence of control hamiltonians
ctrl_hamil = [(H_d + H_c[j]).full() for j in range(args.qubit_num * 2)]

# initial control
if args.initial_type == "ave":
    initial = np.ones(num_switch + 1) * args.evo_time / (num_switch + 1)
if args.initial_type == "rnd":
    initial_pre = np.random.random(num_switch + 1)
    initial = initial_pre.copy() / sum(initial_pre) * args.evo_time
if args.initial_type == "warm":
    initial = warm_start_length

# build optimizer
spin_opt = SwitchTimeOpt()
min_time = 1
spin_opt.build_optimizer(
    ctrl_hamil, ctrl_hamil_idx, initial, X_0.full(), X_targ.full(), args.evo_time, num_switch, args.min_up_time, None)
start = time.time()
res = spin_opt.optimize()
end = time.time()

if not os.path.exists("../output/SwitchTime/"):
    os.makedirs("../output/SwitchTime/")
if not os.path.exists("../control/SwitchTime/"):
    os.makedirs("../control/SwitchTime/")
if not os.path.exists("../figure/SwitchTime/"):
    os.makedirs("../figure/SwitchTime/")

# output file
output_name = "../output/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name + str(args.qubit_num), str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type,
    str(args.min_up_time)) + ".log"
output_file = open(output_name, "w+")
print(res, file=output_file)
print("switching time points", spin_opt.switch_time, file=output_file)
print("computational time", end - start, file=output_file)

# retrieve control
control_name = "../control/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name + str(args.qubit_num), str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type,
    str(args.min_up_time)) + ".csv"
control = spin_opt.retrieve_control(args.n_ts)
np.savetxt(control_name, control, delimiter=",")

print("alpha", args.alpha, file=output_file)
tv_norm = spin_opt.tv_norm()
print("tv norm", tv_norm, file=output_file)
print("objective with tv norm", spin_opt.obj + args.alpha * tv_norm, file=output_file)

# figure file
figure_name = "../figure/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name + str(args.qubit_num), str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type,
    str(args.min_up_time)) + ".png"
spin_opt.draw_control(figure_name)

b_bin = np.loadtxt(control_name, delimiter=",")
f = open(output_name, "a+")
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()