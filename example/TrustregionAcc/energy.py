import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../..")
from tools import *
from trustregion.trust_region_acc import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='EnergyTR')
# number of qubits
parser.add_argument('--n', help='number of qubits', type=int, default=2)
# number of edges for generating regular graph
parser.add_argument('--num_edges', help='number of edges for generating regular graph', type=int, default=1)
# if generate the graph randomly
parser.add_argument('--rgraph', help='if generate the graph randomly', type=int, default=0)
# number of instances
parser.add_argument('--seed', help='random seed', type=int, default=0)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=2)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=40)
# initial control file for the trust-region method
parser.add_argument('--initial_file', help='file name of initial control', type=str, default=None)
# TV regularizer parameter
parser.add_argument('--alpha', help='TV regularizer parameter', type=float, default=0.001)
# ratio threshold for decrease to adjust trust region
parser.add_argument('--sigma', help='ratio threshold for decrease to adjust trust region', type=float, default=0.25)
# ratio threshold for decrease to update central point
parser.add_argument('--eta', help='ratio threshold for decrease to update central point', type=float, default=0.001)
# threshold for region to start precise search
parser.add_argument('--threshold', help='threshold for region to start precise search', type=int, default=30)
# max iterations for trust-region method
parser.add_argument('--max_iter', help='max iterations for trust-region method', type=int, default=100)
# problem type of trust-region method
parser.add_argument('--tr_type', help='problem type of trust-region method', type=str, default='tv')
# if use hard constraints, the type of hard constraints
parser.add_argument('--hard_type', help='type of hard constraints if use them', type=str, default='minup')
# minimum up time steps
# parser.add_argument('--min_up', help='minimum up time steps', nargs='+', type=int, default=10)
parser.add_argument('--min_up', help='minimum up time steps', type=int, default=10)
# maximum number of switches
parser.add_argument('--max_switch', help='maximum number of switches', type=int, default=10)

args = parser.parse_args()

if args.rgraph == 0:
    Jij, edges = generate_Jij_MC(args.n, args.num_edges, 100)

    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

    args.seed = 0

if args.rgraph == 1:
    Jij = generate_Jij(args.n, args.seed)
    C = get_ham(args.n, True, Jij)
    B = get_ham(args.n, False, Jij)

y0 = uniform(args.n)

if not os.path.exists("../output/TrustregionAcc/"):
    os.makedirs("../output/TrustregionAcc/")
if not os.path.exists("../control/TrustregionAcc/"):
    os.makedirs("../control/TrustregionAcc/")
if not os.path.exists("../figure/TrustregionAcc/"):
    os.makedirs("../figure/TrustregionAcc/")

if args.tr_type in ['tv', 'tvc']:
    output_num = "../output/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".log"
    output_fig = "../figure/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".png"
    output_control = "../control/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".csv"
    tr_optimizer = TrustRegionAcc()
    tr_optimizer.build_optimizer(np.zeros((2 ** args.n, 2 ** args.n)), [B, C], y0[0:2 ** args.n], None, args.n_ts, 
                                 args.evo_time, alpha=args.alpha,
                                 obj_type='energy', initial_file=args.initial_file,
                                 sigma=args.sigma, eta=args.eta, delta_threshold=args.threshold,
                                 max_iter=args.max_iter, out_log_file=output_num, out_control_file=output_control)
    if args.tr_type == 'tv':
        tr_optimizer.trust_region_method_tv(type='binary')
    if args.tr_type == 'tvc':
        tr_optimizer.trust_region_method_tv(type='continuous')


if args.tr_type == 'hard':
    if args.hard_type == 'minup':
        output_num = "../output/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".log"
        output_fig = "../figure/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".png"
        output_control = "../control/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                         "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                             args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".csv"
        cons_parameter = dict(hard_type=args.hard_type, time=args.min_up)
        
    if args.hard_type == "maxswitch":
        output_num = "../output/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.max_switch) + ".log"
        output_fig = "../figure/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.max_switch) + ".png"
        output_control = "../control/TrustregionAcc/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                         "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                             args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type,
                             args.max_switch) + ".csv"
        cons_parameter = dict(hard_type=args.hard_type, switch=args.max_switch)

    tr_optimizer = TrustRegionAcc()
    tr_optimizer.build_optimizer(np.zeros((2 ** args.n, 2 ** args.n)), [B, C], y0[0:2 ** args.n], None, args.n_ts, 
                                 args.evo_time, alpha=args.alpha,
                                 obj_type='energy', initial_file=args.initial_file,
                                 sigma=args.sigma, eta=args.eta, delta_threshold=args.threshold,
                                 max_iter=args.max_iter, out_log_file=output_num, out_control_file=output_control)
    tr_optimizer.trust_region_method_hard(cons_parameter)

b_bin = np.loadtxt(output_control, delimiter=",")
if len(b_bin.shape) == 1:
    b_bin = np.expand_dims(b_bin, axis=1)
fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.ylim([0, 1])
marker_list = ['-o', '--^', '-*', '--s']
marker_size_list = [5, 5, 8, 5]
for j in range(b_bin.shape[1]):
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_bin[:, j], b_bin[-1, j])), marker_list[j],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig(output_fig)
