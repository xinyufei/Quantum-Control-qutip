from pycombina import *
import numpy as np
import matplotlib.pyplot as plt
import time


def rounding(b_rel, type, min_up_times=0, max_switches=10, compare=False, bin=True, out_fig=None):

    t = np.array([t_step for t_step in range(b_rel.shape[0] + 1)])
    ts_interval = 20

    binapprox = BinApprox(t, b_rel)

    start = time.time()
    if type == "SUR":
        min_up_times = "SUR"
        sur = CombinaSUR(binapprox)
        sur.solve()

    if type == "minup":
        # binapprox.set_n_max_switches(n_max_switches=[min_up_times, min_up_times])
        binapprox.set_min_up_times(min_up_times=[min_up_times] * b_rel.shape[1])
        combina = CombinaBnB(binapprox)
        combina.solve()

    if type == "maxswitch":
        binapprox.set_n_max_switches(n_max_switches=[max_switches] * b_rel.shape[1])
        combina = CombinaBnB(binapprox)
        combina.solve()
    end = time.time()

    b_bin = binapprox.b_bin

    if compare and out_fig is not None:
        for control_idx in range(int(np.ceil(b_bin.shape[0] / 2))):
            plt.figure(dpi=300)
            f, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.step(t[:-1] / ts_interval, b_rel[:, control_idx * 2], label="b_rel", color="C0", linestyle="dashed",
                     where="post")
            ax1.step(t[:-1] / ts_interval, b_bin[control_idx * 2, :], label="b_bin", color="C0", where="post")
            ax1.legend(loc="upper left")
            ax1.set_ylabel("u_" + str(control_idx * 2 + 1))
            if control_idx * 2 + 1 < b_bin.shape[0]:
                ax2.step(t[:-1] / ts_interval, b_rel[:, control_idx * 2 + 1], label="b_rel", color="C1",
                         linestyle="dashed", where="post")
                ax2.step(t[:-1] / ts_interval, b_bin[control_idx * 2 + 1, :], label="b_bin", color="C1", where="post")
                ax2.legend(loc="upper left")
                ax2.set_ylabel("u_" + str(control_idx * 2 + 2))
            if type == "SUR":
                plt.savefig(out_fig + "_bvsr_" + str(min_up_times) + "_" + str(control_idx * 2 + 1) + "+" +
                            str(control_idx * 2 + 2) + "_SUR.png")
            if type == "minup":
                plt.savefig(out_fig + "_bvsr_" + str(min_up_times) + "_" + str(control_idx * 2 + 1) + "+" +
                            str(control_idx * 2 + 2) + "_minup" + str(min_up_times) + ".png")
            if type == "maxswitch":
                plt.savefig(out_fig + "_bvsr_" + str(min_up_times) + "_" + str(control_idx * 2 + 1) + "+" +
                            str(control_idx * 2 + 2) + "_maxswitch" + str(max_switches) + ".png")

    if bin and out_fig is not None:
        fig = plt.figure(dpi=300)
        # plt.title("Rounded Optimised Control Sequences")
        plt.xlabel("Time")
        plt.ylabel("Control amplitude")
        plt.ylim([0, 1])
        marker_list = ['-o', '--^', '-*', '--s']
        marker_size_list = [5, 5, 8, 5]
        for j in range(b_rel.shape[1]):
            plt.step(t, np.hstack((b_rel[:, j], b_rel[-1, j])),
                     marker_list[j], where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
                     markersize=marker_size_list[j])
        plt.legend()
        if type == "SUR":
            plt.savefig(out_fig + "_binary_SUR.png")
        if type == "minup":
            plt.savefig(out_fig + "_binary_minup" + str(min_up_times) + ".png")
        if type == "maxswitch":
            plt.savefig(out_fig + "_binary_maxswitch" + str(max_switches) + ".png")

    return b_bin, end - start


# if __name__ == '__main__':
#     file_name = "control/CNOTSUM1_evotime20_n_ts400_ptypeZERO_offset0.5_objUNIT.csv"
#     rounding(file_name, "SUR", 1)
