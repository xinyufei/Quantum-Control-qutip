import os
import argparse
import matplotlib.pyplot as plt
from qutip import identity, sigmax, sigmaz, sigmay, tensor, Qobj
from qutip.qip.operations.gates import cnot

import sys

sys.path.append("../..")
from tools import *
from selfoptcontrol.optcontrol_admm_cnot import Optcontrol_ADMM_CNOT

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='NOTleakADMM')
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
parser.add_argument('--max_iter_admm', help='maximum iterations for ADMM', type=int, default=100)
# maximum computational time
parser.add_argument('--max_time_admm', help='maximum computational time for ADMM', type=float,
                    default=7200 * 200)

args = parser.parse_args()

H_d_origin = Qobj(
    0 * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) + 2 * math.pi * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))
H_c_origin = [Qobj(1 / 2 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) 
                   + np.sqrt(2) / 2 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])),
              Qobj(1 / 2 * np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
                   + np.sqrt(2) / 2 * np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]))]
# print(H_d, H_c)
H_d = H_d_origin - H_c_origin[0] - H_c_origin[1]
H_c = [2 * hc for hc in H_c_origin]
H_d = H_d_origin
H_c = H_c_origin
# start point for the gate evolution
X_0 = identity(3)
# Target for the gate evolution
X_targ = Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
# print(X_targ)
# exit()

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
NOTleak_opt_admm = Optcontrol_ADMM_CNOT()
NOTleak_opt_admm.build_optimizer(H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time,
                                 amp_lbound=0, amp_ubound=1, sum_cons_1=False,
                                 fid_err_targ=args.fid_err_targ, min_grad=args.min_grad,
                                 max_iter_step=args.max_iter_step,
                                 max_wall_time_step=args.max_time_step, fid_type="UNIT", phase_option="PSU",
                                 p_type=args.initial_type, seed=None, constant=args.offset,
                                 initial_control=args.initial_control,
                                 output_num=output_num, output_fig=output_fig, output_control=output_control,
                                 alpha=args.alpha, rho=args.rho,
                                 max_iter_admm=args.max_iter_admm, max_wall_time_admm=args.max_time_admm)
NOTleak_opt_admm.optimize_admm()

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
rel_result = time_evolution(H_d.full(), [hc.full() for hc in H_c], args.n_ts, args.evo_time, b_rel, X_0.full(), False, 1)
print(compute_obj_fid(X_targ, rel_result, phase="PSU", leak=True))
