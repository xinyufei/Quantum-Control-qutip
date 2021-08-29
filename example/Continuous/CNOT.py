import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
sys.path.append("../..")
from selfoptcontrol.optcontrol import optcontrol
from tools.evolution import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='CNOT')
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=10)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=200)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# Fidelity error target
parser.add_argument('--fid_err_targ', help='target for the fidelity error', type=float, default=1e-10)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter', help='maximum number of iterations', type=int, default=500)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time', help='maximum allowed computational time (seconds)', type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-8)

args = parser.parse_args()
# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate with summation one constraint
# The control Hamiltonians (Qobj classes)
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()

print(H_d.full().conj().T.dot(H_d.full()))
# objective value type
obj_type = "UNIT"

if not os.path.exists("../output/Continuous/"):
    os.makedirs("../output/Continuous/")
if not os.path.exists("../control/Continuous/"):
    os.makedirs("../control/Continuous/")
if not os.path.exists("../figure/Continuous/"):
    os.makedirs("../figure/Continuous/")

output_num = "../output/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type) + ".log"
output_fig = "../figure/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type) + ".png"
output_control = "../control/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type) + ".csv"

# solve the optimization model
optcontrol(args.name, H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time, args.initial_type, args.initial_control,
           output_num, output_fig, output_control, fid_err_targ=args.fid_err_targ, max_iter=args.max_iter,
           max_wall_time=args.max_time, min_grad=args.min_grad, constant=args.offset)

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
    # plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])),
    #          where='post', linewidth=2, label='controller ' + str(j + 1))
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig(output_fig.split(".png")[0] + "_continuous.png")

f = open(output_num, "a+")
print("total tv norm", compute_TV_norm(b_rel), file=f)
