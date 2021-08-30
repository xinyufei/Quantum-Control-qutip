from pycombina import *
import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gb


class Rounding:

    def __init__(self):
        self.b_rel = None
        self.b_bin = None
        self.evo_time = None
        self.time_steps = None
        self.type = "SUR"
        self.min_up_times = 0
        self.max_switches = 10
        self.compare = False
        self.bin = False
        self.out_fig = None
        self.t = None
        self.n_ctrls = None
        self.delta_t = None

    def build_rounding_optimizer(self, b_rel, evo_time, time_steps, type, min_up_times=0, max_switches=10,
                                 compare=False, bin=True, out_fig=None):
        self.b_rel = b_rel
        self.evo_time = evo_time
        self.time_steps = time_steps
        self.type = type
        self.min_up_times = min_up_times
        self.max_switches = max_switches
        self.compare = compare
        self.bin = bin
        self.out_fig = out_fig

        self.n_ctrls = self.b_rel.shape[1]
        self.t = np.linspace(0, self.evo_time, self.time_steps + 1)
        self.delta_t = self.evo_time / self.time_steps

    def draw_compare_figure(self):
        if self.out_fig:
            for control_idx in range(int(np.ceil(self.time_steps / 2))):
                plt.figure(dpi=300)
                f, (ax1, ax2) = plt.subplots(2, sharex=True)
                ax1.step(self.t, self.b_rel[:, control_idx * 2], label="b_rel", color="C0",
                         linestyle="dashed", where="post")
                ax1.step(self.t, self.b_bin[:, control_idx * 2], label="b_bin", color="C0",
                         where="post")
                ax1.legend(loc="upper left")
                ax1.set_ylabel("u_" + str(control_idx * 2 + 1))
                if control_idx * 2 + 1 < self.n_ctrls:
                    ax2.step(self.t, self.b_rel[:, control_idx * 2 + 1], label="b_rel", color="C1",
                             linestyle="dashed", where="post")
                    ax2.step(self.t, self.b_bin[:, control_idx * 2 + 1], label="b_bin", color="C1",
                             where="post")
                    ax2.legend(loc="upper left")
                    ax2.set_ylabel("u_" + str(control_idx * 2 + 2))
                if self.type == "SUR":
                    plt.savefig(self.out_fig + "_bvsr_" + str(control_idx * 2 + 1) + "+" + str(control_idx * 2 + 2) +
                                "_SUR.png")
                if self.type == "minup":
                    plt.savefig(self.out_fig + "_bvsr_" + str(control_idx * 2 + 1) + "+" + str(control_idx * 2 + 2) +
                                "_minup" + str(self.min_up_times) + ".png")
                if self.type == "maxswitch":
                    plt.savefig(self.out_fig + "_bvsr_" + str(control_idx * 2 + 1) + "+" + str(control_idx * 2 + 2) +
                                "_maxswitch" + str(self.max_switches) + ".png")

    def draw_bin_figure(self):
        if self.out_fig:
            fig = plt.figure(dpi=300)
            # plt.title("Rounded Optimised Control Sequences")
            plt.xlabel("Time")
            plt.ylabel("Control amplitude")
            plt.ylim([0, 1])
            marker_list = ['-o', '--^', '-*', '--s']
            marker_size_list = [5, 5, 8, 5]
            for j in range(self.n_ctrls):
                plt.step(self.t, np.hstack((self.b_bin[:, j], self.b_bin[-1, j])),
                         marker_list[j], where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
                         markersize=marker_size_list[j])
            plt.legend()
            if self.type == "SUR":
                plt.savefig(self.out_fig + "_binary_SUR.png")
            if self.type == "minup":
                plt.savefig(self.out_fig + "_binary_minup" + str(self.min_up_times) + ".png")
            if self.type == "maxswitch":
                plt.savefig(self.out_fig + "_binary_maxswitch" + str(self.max_switches) + ".png")

    def rounding_with_sos1(self):

        binapprox = BinApprox(self.t, self.b_rel)

        start = time.time()
        if self.type == "SUR":
            self.min_up_times = "SUR"
            sur = CombinaSUR(binapprox)
            sur.solve()

        if self.type == "minup":
            # binapprox.set_n_max_switches(n_max_switches=[self.min_up_times, self.min_up_times])
            binapprox.set_min_up_times(
                min_up_times=[self.min_up_times * self.evo_time / self.time_steps] * self.n_ctrls)
            combina = CombinaBnB(binapprox)
            combina.solve()

        if self.type == "maxswitch":
            binapprox.set_n_max_switches(n_max_switches=[self.max_switches] * self.n_ctrls)
            combina = CombinaBnB(binapprox)
            combina.solve()
        end = time.time()

        self.b_bin = binapprox.b_bin.T

        if self.compare:
            self.draw_compare_figure()

        if self.bin:
            self.draw_bin_figure()

        return self.b_bin, end - start

    def rounding_without_sos1(self):

        start = time.time()

        round = gb.Model()
        bin_val = round.addVars(self.time_steps, self.n_ctrls, vtype=gb.GRB.BINARY)
        up_diff = round.addVar(lb=0)

        round.addConstrs(gb.quicksum(self.b_rel[t, j] - bin_val[t, j] for t in range(k)) * self.delta_t + up_diff >= 0
                         for j in range(self.n_ctrls) for k in range(1, self.time_steps + 1))
        round.addConstrs(gb.quicksum(self.b_rel[t, j] - bin_val[t, j] for t in range(k)) * self.delta_t - up_diff <= 0
                         for j in range(self.n_ctrls) for k in range(1, self.time_steps + 1))

        round.setObjective(up_diff)
        round.optimize()

        end = time.time()

        self.b_bin = np.zeros((self.time_steps, self.n_ctrls))
        for j in range(self.n_ctrls):
            for k in range(self.time_steps):
                self.b_bin[k, j] = bin_val[k, j].x

        if self.compare:
            self.draw_compare_figure()

        if self.bin:
            self.draw_bin_figure()

        return self.b_bin, end - start


# if __name__ == '__main__':
#     file_name = "control/CNOTSUM1_evotime20_n_ts400_ptypeZERO_offset0.5_objUNIT.csv"
#     rounding(file_name, "SUR", 1)
