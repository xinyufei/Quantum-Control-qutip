import os
import argparse
import matplotlib.pyplot as plt
from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

import sys
sys.path.append("../..")
from tools import *
from selfoptcontrol.optcontrol_admm_cnot import Optcontrol_ADMM_CNOT

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='CNOTADMMSOS1')
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=1)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=20)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# Fidelity error target
parser.add_argument('--fid_err_targ', help='target for the fidelity error', type=float, default=1e-10)
# Maximum iterations for the GRAPE algorithm at each step
parser.add_argument('--max_iter_step', help='maximum number of iterations for the GRAPE algorithm at each step',
                    type=int, default=1000)
# Maximum (elapsed) time allowed in seconds for each step
parser.add_argument('--max_time_step', help='maximum allowed computational time (seconds)for each step',
                    type=float, default=7200)
# Minimum gradient (sum of gradients squared) for each step
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient for each step', type=float, default=1e-6)
# TV regularizer parameter
parser.add_argument('--alpha', help='TV regularizer parameter', type=float, default=0.001)
# Lagrangian penalty parameter
parser.add_argument('--rho', help='Lagrangian penalty function parameter', type=float, default=0.25)
# maximum iterations for ADMM
parser.add_argument('--max_iter_admm', help='maximum iterations for ADMM', type=int, default=200)
# maximum computational time
parser.add_argument('--max_time_admm', help='maximum computational time for ADMM', type=float,
                    default=7200 * 200)

args = parser.parse_args()

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


if not os.path.exists("../output/ADMM/"):
    os.makedirs("../output/ADMM/")
if not os.path.exists("../control/ADMM/"):
    os.makedirs("../control/ADMM/")
if not os.path.exists("../figure/ADMM/"):
    os.makedirs("../figure/ADMM/")

output_num = "../output/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type, args.alpha, args.rho,
    args.max_iter_admm) + ".log"
output_fig = "../figure/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type, args.alpha, args.rho,
    args.max_iter_admm) + ".png"
output_control = "../control/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
    args.name, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type, args.alpha, args.rho,
    args.max_iter_admm) + ".csv"

# solve the optimization model
CNOT_opt_admm = Optcontrol_ADMM_CNOT()
CNOT_opt_admm.build_optimizer(H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time,
                              amp_lbound=0, amp_ubound=1, sum_cons_1=True,
                              fid_err_targ=args.fid_err_targ, min_grad=args.min_grad, max_iter_step=args.max_iter_step,
                              max_wall_time_step=args.max_time_step, fid_type="UNIT", phase_option="PSU",
                              p_type=args.initial_type, seed=None, constant=args.offset,
                              initial_control=args.initial_control,
                              output_num=output_num, output_fig=output_fig, output_control=output_control,
                              alpha=args.alpha, rho=args.rho,
                              max_iter_admm=args.max_iter_admm, max_wall_time_admm=args.max_time_admm)
CNOT_opt_admm.optimize_admm()


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
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig(output_fig.split(".png")[0] + "_continuous.png")
