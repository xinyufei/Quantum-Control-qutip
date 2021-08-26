from pycombina import *
import numpy as np
import matplotlib.pyplot as plt


def rounding(file_name, type, min_up_times=0):
    file = open(file_name)
    b_rel = np.loadtxt(file, delimiter=",")
    t = np.array([t_step for t_step in range(b_rel.shape[0] + 1)])
    ts_interval = 20

    binapprox = BinApprox(t, b_rel)

    if type == "SUR":
        min_up_times = "SUR"
        sur = CombinaSUR(binapprox)
        sur.solve()

    if type == "BnB":
        # binapprox.set_n_max_switches(n_max_switches=[min_up_times, min_up_times])
        binapprox.set_min_up_times(min_up_times=[min_up_times] * b_rel.shape[1])
        combina = CombinaBnB(binapprox)
        combina.solve()

    b_bin = binapprox.b_bin

    for control_idx in range(int(np.ceil(b_bin.shape[0] / 2))):
        f, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.step(t[:-1] / ts_interval, b_rel[:, control_idx * 2], label="b_rel", color="C0", linestyle="dashed", where="post")
        ax1.step(t[:-1] / ts_interval, b_bin[control_idx * 2, :], label="b_bin", color="C0", where="post")
        ax1.legend(loc="upper left")
        ax1.set_ylabel("u_" + str(control_idx * 2 + 1))
        if control_idx * 2 + 1 < b_bin.shape[0]:
            ax2.step(t[:-1] / ts_interval, b_rel[:, control_idx * 2 + 1], label="b_rel", color="C1", linestyle="dashed", where="post")
            ax2.step(t[:-1] / ts_interval, b_bin[control_idx * 2 + 1, :], label="b_bin", color="C1", where="post")
            ax2.legend(loc="upper left")
            ax2.set_ylabel("u_" + str(control_idx * 2 + 2))
        plt.savefig(file_name.split(".csv")[0] + "_bvsr_" + str(min_up_times) + "_" + str(control_idx * 2 + 1) + "+" +
                    str(control_idx * 2 + 2) + ".png")

    fig = plt.figure()
    plt.title("Rounded Optimised Control Sequences")
    plt.xlabel("Time")
    plt.ylabel("Control amplitude")
    for j in range(b_rel.shape[1]):
        plt.step(t, np.hstack((b_bin[j, :], b_bin[j, -1])),
                 where='post')
    plt.savefig(file_name.split(".csv")[0] + "_binary_" + str(min_up_times) + ".png")

    return b_bin


if __name__ == '__main__':
    file_name = "control/CNOTSUM1_evotime20_n_ts400_ptypeZERO_offset0.5_objUNIT.csv"
    rounding(file_name, "SUR", 1)
