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
parser.add_argument('--name', help='example name', type=str, default='Molecule')
# name of molecule
parser.add_argument('--molecule', help='molecule name', type=str, default='H2')
# number of quantum bits
parser.add_argument('--qubit_num', help='number of quantum bits', type=int, default=2)
# evolution time
parser.add_argument('--evo_time', help='evolution time', type=float, default=5)
# time steps
parser.add_argument('--n_ts', help='time steps', type=int, default=100)
# initial type
parser.add_argument('--initial_type', help='initial controls type', type=str, default='CONSTANT')
# initial constant value
parser.add_argument('--offset', help='initial constant value', type=float, default=0.5)
# initial control file
parser.add_argument('--initial_control', help='file name of initial control', type=str, default=None)
# penalty parameter for SOS1 property
parser.add_argument('--sum_penalty', help='penalty parameter for L_2 term', type=float, default=0)
# Fidelity error target
parser.add_argument('--fid_err_targ', help='target for the fidelity error', type=float, default=1e-8)
# Maximum iterations for the optimise algorithm
parser.add_argument('--max_iter', help='maximum number of iterations', type=int, default=3000)
# Maximum (elapsed) time allowed in seconds
parser.add_argument('--max_time', help='maximum allowed computational time (seconds)', type=float, default=7200)
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
parser.add_argument('--min_grad', help='minimum gradient', type=float, default=1e-6)
# indicator to generate target file
parser.add_argument('--gen_target', help='indicator to generate target file', type=int, default=0)
# file store the target circuit
parser.add_argument('--target', help='unitary matrix of target circuit', type=str, default=None)

args = parser.parse_args()

# args.name="MoleculeNew2"
# args.qubit_num=4
# args.molecule="LiH"
# args.evo_time=20
# args.n_ts=200
# args.target="../control/Continuous/MoleculeNew2_LiH_evotime20.0_n_ts200_target.csv"

d = 2
Hops, H0, U0, U = generate_molecule_func(args.qubit_num, d, args.molecule, optimize=True)

if args.target is not None:
    U = np.loadtxt(args.target, dtype=np.complex_, delimiter=',')
elif args.gen_target == 1:
    np.savetxt("../control/Continuous/" + "{}_evotime{}_n_ts{}".format(
        args.name + "_" + args.molecule, args.evo_time, args.n_ts) + "_target.csv", U, delimiter=",")
else:
    print("Please provide the target file!")
    exit()

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

if not os.path.exists("../output/Continuous/"):
    os.makedirs("../output/Continuous/")
if not os.path.exists("../control/Continuous/"):
    os.makedirs("../control/Continuous/")
if not os.path.exists("../figure/Continuous/"):
    os.makedirs("../figure/Continuous/")

output_num = "../output/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_sum_penalty{}".format(
    args.name + "_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type,
    args.sum_penalty) + ".log"
output_fig = "../figure/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_sum_penalty{}".format(
    args.name + "_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type,
    args.sum_penalty) + ".png"
output_control = "../control/Continuous/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_sum_penalty{}".format(
    args.name + "_" + args.molecule, args.evo_time, args.n_ts, args.initial_type, args.offset, obj_type,
    args.sum_penalty) + ".csv"

# solve the optimization model
ops_max_amp = 1
Hadamard_penalized = Optcontrol_Penalized_Qutip()
Hadamard_penalized.build_optimizer(H_d, H_c, X_0, X_targ, args.n_ts, args.evo_time,
                                   amp_lbound=0, amp_ubound=1, ops_max_amp=ops_max_amp,
                                   fid_err_targ=args.fid_err_targ, min_grad=args.min_grad,
                                   max_wall_time_step=args.max_time, max_iter_step=args.max_iter,
                                   fid_type="UNIT", phase_option="PSU",
                                   p_type=args.initial_type, seed=None,
                                   constant=args.offset, initial_control=args.initial_control,
                                   output_num=output_num, output_fig=output_fig, output_control=output_control,
                                   penalty=args.sum_penalty, max_controllers=max_controllers)
Hadamard_penalized.optimize_penalized()

# output_control = "../control/Continuous/MoleculeNew2_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv"
b_rel = np.loadtxt(output_control, delimiter=",")
if len(b_rel.shape) == 1:
    b_rel = np.expand_dims(b_rel, axis=1)

# bin_result = time_evolution(H_d.full(), [hc.full() for hc in H_c], args.n_ts, args.evo_time, b_rel, X_0.full(), False,
#                             1)
# print(compute_obj_fid(X_targ, bin_result))

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

f = open(output_num, "a+")
print("total tv norm", compute_TV_norm(b_rel), file=f)
print("total l2 norm", compute_sum_cons(b_rel, 1), file=f)
f.close()
