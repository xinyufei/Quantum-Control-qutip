import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.append("../..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from tools import *
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from selfoptcontrol.optcontrol_penalized_qutip import Optcontrol_Penalized_Qutip

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='Hadamard')
# number of quantum bits
parser.add_argument('--qubit_num', help='number of quantum bits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=8)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=80)
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

# args.n_ts = int(args.qubit_num * 40)
# args.evo_time = args.qubit_num * 4

Hops, H0, U0, U = generate_spin_func(args.qubit_num)

# The control Hamiltonians (Qobj classes)
H_c = [Qobj(hops) for hops in Hops]
# Drift Hamiltonian
H_d = Qobj(H0)
# start point for the gate evolution
X_0 = Qobj(U0)
# Target for the gate evolution
X_targ = Qobj(U)

max_controllers = 1
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

bin_result = time_evolution(np.zeros(2**args.n), [B, C], args.n_ts, args.evo_time, b_bin.T, y0, False, 1)

f = open(output_num, "a+")
print("original objective", compute_obj_energy(C, bin_result), file=f)
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()

np.savetxt(output_control, b_bin.T, delimeter=',')