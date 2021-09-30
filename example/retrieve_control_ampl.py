import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from tools.evolution import *

initial_file_name = "Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_muscod.csv"

initial_control = np.loadtxt("control/Continuous/" + initial_file_name)

final_control = np.zeros((len(initial_control), 2))
final_control[:, 0] = initial_control
final_control[:, 1] = 1 - initial_control

np.savetxt("control/Continuous/" + initial_file_name, final_control, delimiter=',')

evo_time = 2
n_ts = 40

final_control = np.loadtxt("control/Continuous/" + initial_file_name, delimiter=",")
if len(final_control.shape) == 1:
    final_control = np.expand_dims(final_control, axis=1)
fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
plt.xlabel("Time")
plt.ylabel("Control amplitude")
plt.ylim([0, 1])
marker_list = ['-o', '--^', '-*', '--s']
marker_size_list = [5, 5, 8, 5]
for j in range(final_control.shape[1]):
    plt.step(np.linspace(0, evo_time, n_ts + 1), np.hstack((final_control[:, j], final_control[-1, j])),
             marker_list[j], where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
             markersize=marker_size_list[j])
plt.legend()
plt.savefig("figure/Continuous/" + initial_file_name.split(".csv")[0] + "_continuous.png")

f = open("output/Continuous/" + initial_file_name.split(".csv")[0] + ".log", "w+")
print("total tv norm", compute_TV_norm(final_control), file=f)
