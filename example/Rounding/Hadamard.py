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
# if initial sos1 property holds
parser.add_argument('--sos1', help='sos1 property holds or not', type=int, default=1)
# if target sos1 property holds
parser.add_argument('--t_sos1', help='sos1 property holds or not', type=int, default=1)
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
if args.sos1:
    b_bin, c_time = round.rounding_with_sos1()
else:
    b_bin, c_time = round.rounding_without_sos1(sos1=args.t_sos1)

bin_result = time_evolution(H_d.full(), [hc.full() for hc in H_c], args.n_ts, args.evo_time, b_bin, X_0.full(), False,
                            1)

f = open(output_num, "w+")
print("computational time", c_time, file=f)
print("original objective", compute_obj_fid(X_targ, bin_result), file=f)
print("total tv norm", compute_TV_norm(b_bin), file=f)
f.close()

np.savetxt(output_control, b_bin, delimiter=',')