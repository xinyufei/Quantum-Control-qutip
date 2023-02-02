import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../..")
from tools.auxiliary_energy import *
from tools.evolution import *
from selfoptcontrol.optcontrol_energy import optcontrol_energy

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='Energy')
# number of qubits
parser.add_argument('--n', help='number of qubits', type=int, default=2)
# number of edges for generating regular graph
parser.add_argument('--num_edges', help='number of edges for generating regular graph', type=int,
                    default=1)
# if generate the graph randomly
parser.add_argument('--rgraph', help='if generate the graph randomly', type=int, default=0)
# seed to generate random graph
parser.add_argument('--seed', help='random seed', type=int, default=0)
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
parser.add_argument('--max_iter', help='maximum number of iterations', type=int, default=100)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time', help='maximum allowed computational time (seconds)', type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-6)

args = parser.parse_args()

y0 = uniform(args.n)

if not os.path.exists("../output/Continuous/"):
    os.makedirs("../output/Continuous/")
if not os.path.exists("../control/Continuous/"):
    os.makedirs("../control/Continuous/")
if not os.path.exists("../figure/Continuous/"):
    os.makedirs("../figure/Continuous/")

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)
    
    args.seed = 0
    

output_num = "../output/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.seed) + ".log"
output_fig = "../figure/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.seed) + ".png"
output_control = "../control/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_instance{}".format(
    args.name + str(args.n), args.evo_time, args.n_ts, args.initial_type, args.offset, args.seed) + ".csv"
    
if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)
    
opt = optcontrol_energy()
opt.build_optimizer(B, C, args.n, y0[0:2**args.n], args.n_ts, args.evo_time, args.initial_type,
                    output_fig=output_fig, output_num=output_num, output_control=output_control,
                    max_iter=args.max_iter, max_wall_time=args.max_time, min_grad=args.min_grad, 
                    constant=args.offset)

opt.optimize()

b_rel = np.loadtxt(output_control, delimiter=",")
# b_rel = np.loadtxt("../control/Rounding/Energy6_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_instance1_1_SUR.csv", delimiter=',')
# print(opt._compute_energy(b_rel[:, 0]))
bin_result = time_evolution(np.zeros((2**args.n, 2**args.n), dtype=complex), [B, C], args.n_ts, args.evo_time, 
                            b_rel, y0[0:2 ** args.n], False, 1)
if len(b_rel.shape) == 1:
    b_rel = np.expand_dims(b_rel, axis=1)
fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.ylim([0, 1])
marker_list = ['-o', '--^', '-*', '--s']
marker_size_list = [5, 5, 8, 5]
for j in range(b_rel.shape[1]):
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_rel[:, j], b_rel[-1, j])), marker_list[j],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig(output_fig.split(".png")[0] + "_continuous.png")

f = open(output_num, "a+")
print("total tv norm", compute_TV_norm(b_rel), file=f)
print("true energy", min(get_diag(Jij)), file=f)
print("real energy", compute_obj_energy(C, bin_result), file=f)
