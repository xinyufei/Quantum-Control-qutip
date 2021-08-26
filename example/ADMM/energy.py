import argparse
import os
import sys

import matplotlib.pyplot as plt
sys.path.append("../..")
from tools import *
from selfoptcontrol.optcontrol_admm_energy import optcontrol_admm_energy

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='EnergyADMM')
# number of qubits
parser.add_argument('--n', help='number of qubits', type=int, default=2)
# number of edges for generating regular graph
parser.add_argument('--num_edges', help='number of edges for generating regular graph', type=int, default=1)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=2)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=40)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter_step', help='maximum number of iterations', type=int, default=100)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time_step', help='maximum allowed computational time (seconds) for each step',
                    type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient for each step', type=float, default=1e-6)
# TV regularizer parameter
parser.add_argument('--alpha', help='TV regularizer parameter', type=float, default=0.001)
# Lagrangian penalty parameter
parser.add_argument('--rho', help='Lagrangian penalty function parameter', type=float, default=0.01)
# maximum iterations for ADMM
parser.add_argument('--max_iter_admm', help='maximum iterations for ADMM', type=int, default=50)
# maximum computational time
parser.add_argument('--max_time_admm', help='maximum computational time for ADMM', type=float, default=7200*50)

args = parser.parse_args()

Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

C = get_ham(args.n, True, Jij)
B = get_ham(args.n, False, Jij)

y0 = uniform(args.n)

if not os.path.exists("../output/ADMM/"):
    os.makedirs("../output/ADMM/")
if not os.path.exists("../control/ADMM/"):
    os.makedirs("../control/ADMM/")
if not os.path.exists("../figure/ADMM/"):
    os.makedirs("../figure/ADMM/")

output_num = "../output/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_ADMM_{}_iter{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha, args.rho,
    args.max_iter_admm) + ".log"
output_fig = "../figure/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_ADMM_{}_iter{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha, args.rho,
    args.max_iter_admm) + ".png"
output_control = "../control/ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_penalty{}_ADMM_{}_iter{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.alpha, args.rho,
    args.max_iter_admm) + ".csv"


opt = optcontrol_admm_energy()
opt.build_optimizer(B, C, args.n, y0[0:2**args.n], args.n_ts, args.evo_time, initial_type=args.initial_type,
                    initial_control=args.initial_control,
                    output_fig=output_fig, output_num=output_num, output_control=output_control,
                    max_iter=args.max_iter_step, max_wall_time=args.max_time_step, min_grad=args.min_grad,
                    constant=args.offset, rho=args.rho, alpha=args.alpha, max_iter_admm=args.max_iter_admm,
                    max_wall_time_admm=args.max_time_admm)

opt.optimize_admm()

b_rel = np.loadtxt(output_control, delimiter=",")

# state = [y0[0:2**args.n]]
# for k in range(args.n_ts):
#     cur_state = expm(-1j * (b_rel[k, 0] * B + b_rel[k, 1] * C) * args.evo_time/args.n_ts).dot(state[k])
#     state.append(cur_state)
# final_state = state[-1]
#
# print(np.real(final_state.conj().T.dot(C.dot(final_state))))
#
# print(compute_energy_u(opt.tlist, args.evo_time, b_rel[:,0]))


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
