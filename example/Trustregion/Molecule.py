import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

sys.path.append("../..")
from tools import *
from trustregion.trust_region import *

parser = argparse.ArgumentParser()
# name of example
parser.add_argument('--name', help='example name', type=str, default='HadamardTR')
# name of molecule
parser.add_argument('--molecule', help='molecule name', type=str, default='H2')
# number of quantum bits
parser.add_argument('--qubit_num', help='number of quantum bits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=4)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=80)
# sum penalty
parser.add_argument('--sum_penalty', help='penalty for sos1 constraint', type=float, default=0.01)
# initial control file for the trust-region method
parser.add_argument('--initial_file', help='file name of initial control', type=str, default=None)
# if sos1 property holds
parser.add_argument('--sos1', help='sos1 property holds or not', type=int, default=1)
# TV regularizer parameter
parser.add_argument('--alpha', help='TV regularizer parameter', type=float, default=0.0001)
# ratio threshold for decrease to adjust trust region
parser.add_argument('--sigma', help='ratio threshold for decrease to adjust trust region', type=float, default=0.25)
# ratio threshold for decrease to update central point
parser.add_argument('--eta', help='ratio threshold for decrease to update central point', type=float, default=0.001)
# threshold for region to start precise search
parser.add_argument('--threshold', help='threshold for region to start precise search', type=int, default=30)
# max iterations for trust-region method
parser.add_argument('--max_iter', help='max iterations for trust-region method', type=int, default=100)
# problem type of trust-region method
parser.add_argument('--tr_type', help='problem type of trust-region method', type=str, default='tvc')
# if use hard constraints, the type of hard constraints
parser.add_argument('--hard_type', help='type of hard constraints if use them', type=str, default='minup')
# minimum up time steps
# parser.add_argument('--min_up', help='minimum up time steps', nargs='+', type=int, default=10)
parser.add_argument('--min_up', help='minimum up time steps', type=int, default=10)
# maximum number of switches
parser.add_argument('--max_switch', help='maximum number of switches', type=int, default=10)
# file store the target circuit
parser.add_argument('--target', help='unitary matrix of target circuit', type=str, default=None)

args = parser.parse_args()

d = 2

# args.molecule = "LiH"
# args.qubit_num = 4
# args.target = "../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv"
# args.alpha = 0.0001


Hops, H0, U0, U = generate_molecule_func(args.qubit_num, d, args.molecule)
if args.target is not None:
    U = np.loadtxt(args.target, dtype=np.complex_, delimiter=',')
else:
    print("Please provide the target file!")
    exit()
# args.n_ts = 200
# args.evo_time = 20

# args.tr_type = "tvc"
# args.hard_type = "maxswitch"
# args.max_switch = 30

# args.initial_file = "../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv"

if not os.path.exists("../output/Trustregion/"):
    os.makedirs("../output/Trustregion/")
if not os.path.exists("../control/Trustregion/"):
    os.makedirs("../control/Trustregion/")
if not os.path.exists("../figure/Trustregion/"):
    os.makedirs("../figure/Trustregion/")

if args.tr_type in ['tv', 'tvc']:
    output_num = "../output/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".log"
    output_fig = "../figure/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".png"
    output_control = "../control/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[
        0] + "_alpha{}_sigma{}_eta{}_threshold{}_iter{}_type{}".format(args.alpha, args.sigma, args.eta,
                                                                       args.threshold, args.max_iter,
                                                                       args.tr_type) + ".csv"
    tr_optimizer = TrustRegion()
    tr_optimizer.build_optimizer(H0, Hops, U0, U, args.n_ts, args.evo_time, alpha=args.alpha, obj_type='fid',
                                 initial_file=args.initial_file,
                                 sigma=args.sigma, eta=args.eta, delta_threshold=args.threshold,
                                 max_iter=args.max_iter, out_log_file=output_num, out_control_file=output_control)
    if args.tr_type == 'tv':
        tr_optimizer.trust_region_method_tv(sos1=args.sos1, type='binary')
    if args.tr_type == 'tvc':
        # tr_optimizer.trust_region_method_tv(sos1=args.sos1, type='continuous')
        print("call function with l2")
        tr_optimizer.trust_region_method_l2_tv(args.sum_penalty)

if args.tr_type == 'hard':
    if args.hard_type == 'minup':
        output_num = "../output/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".log"
        output_fig = "../figure/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".png"
        output_control = "../control/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                         "_sigma{}_eta{}_threshold{}_iter{}_type{}_time{}".format(
                             args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.min_up) + ".csv"
        cons_parameter = dict(hard_type=args.hard_type, time=args.min_up)
        
    if args.hard_type == "maxswitch":
        output_num = "../output/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.max_switch) + ".log"
        output_fig = "../figure/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                     "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                         args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type, args.max_switch) + ".png"
        output_control = "../control/Trustregion/" + args.initial_file.split('/')[-1].split('.csv')[0] + \
                         "_sigma{}_eta{}_threshold{}_iter{}_type{}_switch{}".format(
                             args.sigma, args.eta, args.threshold, args.max_iter, args.hard_type,
                             args.max_switch) + ".csv"
        cons_parameter = dict(hard_type=args.hard_type, switch=args.max_switch)

    tr_optimizer = TrustRegion()
    tr_optimizer.build_optimizer(H0, Hops, U0, U, args.n_ts, args.evo_time, alpha=args.alpha, obj_type='fid',
                                 initial_file=args.initial_file,
                                 sigma=args.sigma, eta=args.eta, delta_threshold=args.threshold,
                                 max_iter=args.max_iter, out_log_file=output_num, out_control_file=output_control)
    tr_optimizer.trust_region_method_hard(cons_parameter, sos1=args.sos1)

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
    plt.step(np.linspace(0, args.evo_time, args.n_ts + 1), np.hstack((b_bin[:, j], b_bin[-1, j])), marker_list[j % 4],
             where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j % 4])
plt.legend()
plt.savefig(output_fig)
