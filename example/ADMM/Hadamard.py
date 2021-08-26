import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("../..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from selfoptcontrol.optcontrol_all_penalty_qutip import Optcontrol_General_Qutip
from qutip import Qobj
from tools import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='HadamardADMM')
# number of quantum bits
parser.add_argument('--qubit_num', help='number of quantum bits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=8)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=80)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# penalty parameter for SOS1 property
parser.add_argument('--sum_penalty', help='penalty parameter for L_2 term', type=float, default=0)
# Fidelity error target
parser.add_argument('--fid_err_targ', help='target for the fidelity error', type=float, default=1e-10)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter_step', help='maximum number of iterations', type=int, default=1000)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time_step', help='maximum allowed computational time (seconds) for each step',
                    type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient for each step', type=float, default=1e-6)
# TV regularizer parameter
parser.add_argument('--alpha', help='TV regularizer parameter', type=float, default=0.001)
# Lagrangian penalty parameter
parser.add_argument('--rho', help='Lagrangian penalty function parameter', type=float, default=1)
# maximum iterations for ADMM
parser.add_argument('--max_iter_admm', help='maximum iterations for ADMM', type=int, default=200)
# maximum computational time
parser.add_argument('--max_time_admm', help='maximum computational time for ADMM', type=float,
                    default=7200 * 200)

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

obj_type = "UNIT"

if not os.path.exists("../output/ADMM/"):
    os.makedirs("../output/ADMM/")
if not os.path.exists("../control/ADMM/"):
    os.makedirs("../control/ADMM/")
if not os.path.exists("../figure/ADMM/"):
    os.makedirs("../figure/ADMM/")

output_num = "../output/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_sum_penalty{}_rho{}_iter{}".format(
    args.name + str(args.qubit_num), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha,
    args.sum_penalty, args.rho, args.max_iter_admm) + ".log"
output_fig = "../figure/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_sum_penalty{}_rho{}_iter{}".format(
    args.name + str(args.qubit_num), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha,
    args.sum_penalty, args.rho, args.max_iter_admm) + ".png"
output_control = "../control/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_sum_penalty{}_rho{}_iter{}". \
    format(args.name + str(args.qubit_num), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha,
           args.sum_penalty, args.rho, args.max_iter_admm) + ".csv"

# solve the optimization model
ops_max_amp = 1
Hadamard_general = Optcontrol_General_Qutip()
Hadamard_general.build_optimizer(H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time,
                                 amp_lbound=0, amp_ubound=1, ops_max_amp=ops_max_amp,
                                 fid_err_targ=args.fid_err_targ, min_grad=args.min_grad,
                                 max_wall_time_step=args.max_time_step,
                                 max_iter_step=args.max_iter_step, fid_type="UNIT", phase_option="PSU",
                                 p_type=args.initial_type, seed=None, constant=args.offset,
                                 initial_control=args.initial_control,
                                 output_num=output_num, output_fig=output_fig, output_control=output_control,
                                 penalty=args.sum_penalty, max_controllers=1,
                                 sum_cons_1=False, alpha=args.alpha, rho=args.rho, max_iter_admm=args.max_iter_admm,
                                 max_wall_time_admm=args.max_time_admm)

Hadamard_general.optimize_admm()

b_rel = np.loadtxt(output_control, delimiter=",")
if len(b_rel.shape) == 1:
    b_rel = np.expand_dims(b_rel, axis=1)
fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.ylim([0, 1])
# for j in range(b_rel.shape[1]):
#     plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])),
#              where='post', linewidth=2, label='controller ' + str(j + 1))
marker_list = ['-o', '--^', '-*', '--s']
marker_size_list = [5, 5, 8, 5]
for j in range(b_rel.shape[1]):
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j % 4],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j % 4])
plt.legend()
plt.savefig(output_fig.split(".png")[0] + "_continuous.png")
