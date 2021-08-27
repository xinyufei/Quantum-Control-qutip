import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../..")
from tools.auxiliary_energy import *
from tools.evolution import *
from switchingtime.switch_time import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='EnergyST')
# number of qubits
parser.add_argument('--n', help='number of qubits', type=int, default=2)
# number of edges for generating regular graph
parser.add_argument('--num_edges', help='number of edges for generating regular graph', type=int, default=1)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=2)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=40)
# initial type for control variables
parser.add_argument('--initial_type', help='initial type of control variables (rnd, ave, warm)', type=str,
                    default="warm")
# initial control file obtained from ADMM algorithm
parser.add_argument('--admm_control', help='file name of initial control', type=str,
                    default="../control/ADMM/EnergyADMM2_evotime2_n_ts40_ptypeCONSTANT_offset0.5_penalty0.001_ADMM_1_iter5.csv")
# minimum up time constraint
parser.add_argument('--min_up_time', help='minimum up time', type=float, default=0)
# tv regularizer parameter
parser.add_argument('--alpha', help='tv regularizer parameter', type=float, default=0.05)

args = parser.parse_args()

Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

C = get_ham(args.n, True, Jij)
B = get_ham(args.n, False, Jij)

y0 = uniform(args.n)

if args.admm_control is None:
    print("Must provide control results of ADMM!")
    exit()

warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time(args.admm_control, args.n_ts / args.evo_time)

# sequence of control hamiltonians
ctrl_hamil = [B, C]

X_0 = np.expand_dims(y0[0:2**args.n], 1)

if args.initial_type == "ave":
    initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
if args.initial_type == "rnd":
    initial_pre = np.random.random(num_switch + 1)
    initial = initial_pre.copy() / sum(initial_pre) * evo_time
if args.initial_type == "warm":
    initial = warm_start_length

# build optimizer
energy_opt = SwitchTimeOpt()
energy_opt.build_optimizer(
    ctrl_hamil, ctrl_hamil_idx, initial, X_0, None, args.evo_time, num_switch, args.min_up_time, None,
    obj_type='energy')
start = time.time()
res = energy_opt.optimize()
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
print("switching time points", energy_opt.switch_time, file=output_file)
print("computational time", end - start, file=output_file)

# retrieve control
control_name = "../control/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name, str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type, str(args.min_up_time)) + ".csv"
control = energy_opt.retrieve_control(args.n_ts)
np.savetxt(control_name, control, delimiter=",")

print("alpha", args.alpha, file=output_file)
tv_norm = energy_opt.tv_norm()
print("tv norm", tv_norm, file=output_file)
print("objective with tv norm", energy_opt.obj + args.alpha * tv_norm, file=output_file)

# figure file
figure_name = "../figure/SwitchTime/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
    args.name, str(args.evo_time), str(args.n_ts), str(num_switch), args.initial_type, str(args.min_up_time)) + ".png"
energy_opt.draw_control(figure_name)

b_bin = np.loadtxt(control_name, delimiter=",")
f = open(output_name, "a+")
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()