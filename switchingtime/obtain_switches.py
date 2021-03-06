import sys
import time
import numpy as np
from scipy.linalg import expm
from qutip import Qobj
import qutip.control.pulseoptim as cpo
import matplotlib.pyplot as plt

sys.path.append("..")
from tools.auxiliary_energy import *


# from tools.auxiliary_molecule import generate_molecule_func


def obtain_controller(control_arr, thre_max=0.65, thre_min=0.1, seed=None):
    if np.max(control_arr) >= thre_max:
        return np.argmax(control_arr)
    else:
        pool = []
        prob = []
        for j in range(len(control_arr)):
            if control_arr[j] >= thre_min:
                pool.append(j)
                prob.append(control_arr[j])
        prob = np.array(prob) / sum(prob)
        if seed:
            np.random.seed(seed)
        return np.random.choice(pool, 1, p=prob)


class Switches:
    def __init__(self, initial_control_file=None, delta_t=0.05):
        self.initial_control_file = initial_control_file
        self.delta_t = delta_t

        self.j_hat = None
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []

        self.obj_type = None
        self.H_d = None
        self.H_c = None
        self.X_0 = None
        self.X_targ = None
        self.n_ts = None
        self.evo_time = None
        self.optim = None
        self._into = None
        self._onto = None

        self.n_ctrl = None

    def init_gradient_computer(self, H_d, H_c, X_0, X_targ, n_ts, evo_time, obj_type='fid'):
        self.H_d = H_d
        self.H_c = H_c
        self.X_0 = X_0
        self.X_targ = X_targ
        self.n_ts = n_ts
        self.j_hat = np.zeros(n_ts)
        self.evo_time = evo_time
        self.delta_t = self.evo_time / self.n_ts
        self.obj_type = obj_type
        if self.obj_type == 'fid':
            self.optim = cpo.create_pulse_optimizer(Qobj(self.H_d), [Qobj(hc) for hc in self.H_c],
                                                    Qobj(self.X_0), Qobj(self.X_targ), self.n_ts, self.evo_time,
                                                    amp_lbound=0, amp_ubound=1, dyn_type='UNIT', phase_option="PSU",
                                                    init_pulse_params={"offset": 0}, gen_stats=True,
                                                    init_pulse_type="ZERO")
            self.n_ctrl = len(self.H_c)
            
        if self.obj_type == 'energy':
            self.n_ctrl = len(self.H_c) - 1
            self._into = None
            self._onto = None

    def obtain_switches(self, mode='random', thre_min=0.1, thre_max=0.65, thre_ratio=0, seed=None):
        if mode == 'naive':
            return self._obtain_switches_naive()
        if mode == 'random':
            return self._obtain_switches_random(thre_min, thre_max, seed)
        if mode == 'gradient':
            return self._obtain_switches_gradient()
        if mode == 'gradientnu':
            return self._obtain_switches_gradient_noupdate()
        if mode == 'lanu':
            return self._obtain_switches_linear_approximation_noupdate()
        if mode == 'la':
            return self._obtain_switches_linear_approximation(thre_ratio)
        if mode == 'multi':
            return self._obtain_switches_without_sos1(thre_min)
        if mode == 'redu':
            return self._obtain_switches_reduction(thre_ratio)
        if mode == 'sur':
            return self._obtain_switches_sur(thre_ratio)

    def _obtain_switches_without_sos1(self, thre_min):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        num_controllers = initial_control.shape[1]
        self.tau0 = []
        self.num_switches = [0] * num_controllers
        self.hstart = [0] * num_controllers
        cur_state = [0] * num_controllers
        for j in range(num_controllers):
            # self.tau0[j] = []
            prev_switch = 0
            if initial_control[0, j] >= thre_min:
                cur_state[j] = 1
                self.hstart[j] = 1
            for k in range(1, initial_control.shape[0]):
                if initial_control[k, j] >= thre_min:
                    state = 1
                else:
                    state = 0
                if state != cur_state[j]:
                    self.tau0.append(k * self.delta_t - prev_switch)
                    prev_switch = k * self.delta_t
                    self.num_switches[j] += 1
            self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hstart

    def _obtain_switches_naive(self):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        self.j_hat[0] = np.argmax(initial_control[0, :])
        self.hsequence.append(int(self.j_hat[0]))
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            self.j_hat[k] = np.argmax(initial_control[k, :])
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(int(self.j_hat[k]))
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence

    def _obtain_switches_random(self, thre_min, thre_max, seed=None):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        self.j_hat[0] = obtain_controller(initial_control[0, :], thre_max, thre_min, seed)
        self.hsequence.append(int(self.j_hat[0]))
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            self.j_hat[k] = obtain_controller(initial_control[k, :], thre_max, thre_min, seed)
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(int(self.j_hat[k]))
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence

    def _obtain_switches_gradient(self):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        updated_control = initial_control.copy()
        self.grad = np.zeros_like(initial_control)
        # compute gradient
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(updated_control)
        grad = self._compute_gradient(updated_control, -1)
        self.grad[0, :] = grad[0, :].copy()
        # cur_j_hat = int(np.argmin(grad[0, :]))
        cur_j_hat = int(np.argmax(initial_control[0, :]))
        self.j_hat[0] = cur_j_hat
        self.hsequence.append(cur_j_hat)
        updated_control[0, :] = 0
        updated_control[0, cur_j_hat] = 1
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            grad = self._compute_gradient(updated_control, update_idx=k - 1)
            self.grad[k, :] = grad[k, :]
            cur_j_hat = int(np.argmin(grad[k, :]))
            self.j_hat[k] = cur_j_hat
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
            updated_control[k, :] = 0
            updated_control[k, cur_j_hat] = 1
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence

    def _obtain_switches_gradient_noupdate(self):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        updated_control = initial_control.copy()
        # compute gradient
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(updated_control)
        grad = self._compute_gradient(updated_control, -1)
        cur_j_hat = int(np.argmin(grad[0, :]))
        self.j_hat[0] = cur_j_hat
        self.hsequence.append(cur_j_hat)
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            cur_j_hat = int(np.argmin(grad[k, :]))
            self.j_hat[k] = cur_j_hat
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence
    
    def _obtain_switches_linear_approximation_noupdate(self):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        updated_control = initial_control.copy()
        # compute gradient
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(updated_control)
        grad = self._compute_gradient(updated_control, -1)
        cur_j_hat = int(np.argmin(grad[0, :]))
        self.j_hat[0] = cur_j_hat
        self.hsequence.append(cur_j_hat)
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            cur_la = np.zeros_like(grad[k, :])
            for j in range(len(cur_la)):
                cur_la[j] = grad[k, j] - sum(grad[k, :] * updated_control[k, :])            
            cur_j_hat = int(np.argmin(cur_la))
            self.j_hat[k] = cur_j_hat
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence
    
    def _obtain_switches_linear_approximation(self, thre_ratio=0):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")

        # num_control = initial_control.shape[1]
        # initial_control = np.zeros((1600, num_control))
        # # nonzero = [4, 14, 7, 9, 1, 3, 5, 11]
        # nonzero = [4, 10, 7, 9, 1, 3]
        # for idx in nonzero:
        #     initial_control[:, idx] += 1 / len(nonzero)
        # initial_control[:, 4] -= 1 / (2 * len(nonzero) - 1)
        
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        updated_control = initial_control.copy()
        self.la = np.zeros_like(initial_control)
        # compute gradient
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(updated_control)

        # print(self._compute_objective_value(initial_control, -1))

        grad = self._compute_gradient(updated_control, -1)
        for j in range(initial_control.shape[1]):
            self.la[0, j] = grad[0, j] - sum(grad[0, :] * updated_control[0, :])
        cur_j_hat = int(np.argmax(initial_control[0, :]))
        # cur_j_hat = int(np.argmin(self.la[0, :]))
        self.j_hat[0] = cur_j_hat
        self.hsequence.append(cur_j_hat)
        updated_control[0, :] = 0
        updated_control[0, cur_j_hat] = 1
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            grad = self._compute_gradient(updated_control, update_idx=k - 1)
            # cur_la = np.zeros_like(grad[k, :])
            for j in range(initial_control.shape[1]):
                self.la[k, j] = grad[k, j] - sum(grad[k, :] * updated_control[k, :])
            cur_la = self.la[k, int(cur_j_hat)]
            min_la = min(self.la[k, :])
            max_la = max(self.la[k, :])
            if abs(cur_la - min_la) / abs(max_la - min_la) > thre_ratio:
                # cur_j_hat = int(np.argmin(self.la[k, :]))
                cand_cur_j_hat = np.where(self.la[k, :] == np.min(self.la[k, :]))[0]
                cur_j_hat = -1
                # for j_hat in cand_cur_j_hat:
                #     if j_hat not in self.hsequence:
                #         cur_j_hat = j_hat
                #         break
                if cur_j_hat == -1:
                    cur_j_hat = np.argmin(self.la[k, :])        
            self.j_hat[k] = cur_j_hat
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
            updated_control[k, :] = 0
            updated_control[k, cur_j_hat] = 1
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence

    def _obtain_switches_reduction(self, thre_ratio=0):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(initial_control)
        updated_obj = self._compute_objective_value(initial_control, -1)
        updated_control = initial_control.copy()
        cur_control = updated_control.copy()
        prev_switch = 0
        self.cost = np.zeros_like(initial_control)
        for k in range(initial_control.shape[0]):
            cur_control = updated_control.copy()
            for j in range(initial_control.shape[1]):
                cur_control[k, :] = 0
                cur_control[k, j] = 1
                obj = self._compute_objective_value(cur_control, update_idx=-1)
                self.cost[k, j] = obj - updated_obj
            if k == 0:
                cur_j_hat = np.argmin(self.cost[k, :])
            else:
                cur_cost = self.cost[k, int(cur_j_hat)]
                min_cost = min(self.cost[k, :])
                max_cost = max(self.cost[k, :])
                if abs(cur_cost - min_cost) / abs(max_cost - min_cost) > thre_ratio:
                    cur_j_hat = int(np.argmin(self.cost[k, :]))
                # cur_j_hat = np.argmin(self.cost[k, :])
            self.j_hat[k] = cur_j_hat
            if k == 0:
                self.hsequence.append(cur_j_hat)
            elif self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
            updated_control[k, :] = 0
            updated_control[k, cur_j_hat] = 1
            updated_obj = self.cost[k, cur_j_hat] + updated_obj
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence
    
    def _obtain_switches_sur(self, thre_ratio=0):
        initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
        self.num_switches = 0
        self.hsequence = []
        self.tau0 = []
        self.j_hat = np.zeros(initial_control.shape[0])
        self.deviation = np.zeros_like(initial_control)
        self.deviation[0, :] = initial_control[0, :]
        cur_j_hat = np.argmax(self.deviation[0, :])
        self.j_hat[0] = cur_j_hat
        self.hsequence.append(cur_j_hat)
        prev_switch = 0
        for k in range(1, initial_control.shape[0]):
            self.deviation[k, :] = self.deviation[k - 1, :] + initial_control[k, :]
            self.deviation[k, int(cur_j_hat)] = self.deviation[k, int(cur_j_hat)] - 1

            cur_deviation = self.deviation[k, int(cur_j_hat)]
            min_deviation = min(self.deviation[k, :])
            max_deviation = max(self.deviation[k, :])
            if abs(cur_deviation - max_deviation) / abs(max_deviation - min_deviation) > thre_ratio:
            # if cur_deviation / max_deviation < 1 - thre_ratio:
                cur_j_hat = int(np.argmax(self.deviation[k, :]))
            self.j_hat[k] = cur_j_hat
            if self.j_hat[k] != self.j_hat[k - 1]:
                self.num_switches += 1
                self.tau0.append(k * self.delta_t - prev_switch)
                prev_switch = k * self.delta_t
                self.hsequence.append(cur_j_hat)
        self.tau0.append(initial_control.shape[0] * self.delta_t - prev_switch)
        return self.tau0, self.num_switches, self.hsequence
        
    def _time_evolution_energy(self, control_amps, update_idx=-1):
        if update_idx == -1:
            self._into = [self.X_0]
            for k in range(self.n_ts):
                fwd = expm(-1j * (control_amps[k, 0] * self.H_c[0] +
                                  control_amps[k, 1] * self.H_c[1]) * self.delta_t).dot(self._into[k])
                self._into.append(fwd)
        else:
            for k in range(update_idx, self.n_ts):
                fwd = expm(-1j * (control_amps[k, 0] * self.H_c[0] +
                                  control_amps[k, 1] * self.H_c[1]) * self.delta_t).dot(self._into[k])
                self._into[k + 1] = fwd

    def _back_propagation_energy(self, control_amps, update_idx=-1):
        if update_idx == -1:
            self._onto = [self._into[-1].conj().T.dot(self.H_c[1].conj().T)]
            for k in range(self.n_ts):
                bwd = self._onto[k].dot(expm(-1j * (control_amps[self.n_ts - k - 1, 0] * self.H_c[0] +
                                                    control_amps[self.n_ts - k - 1, 1] * self.H_c[1]) * self.delta_t))
                self._onto.append(bwd)
        else:
            for k in range(self.n_ts - update_idx - 1, self.n_ts):
                bwd = self._onto[k].dot(expm(-1j * (control_amps[self.n_ts - k - 1, 0] * self.H_c[0]
                                                    + control_amps[self.n_ts - k - 1, 1] * self.H_c[1]) * self.delta_t))
                self._onto[k + 1] = bwd

    def _compute_gradient(self, control_amps, update_idx=-1):
        if self.obj_type == 'fid':
            grad = self.optim.fid_err_grad_compute(control_amps.reshape(-1))
        if self.obj_type == 'energy':
            # if not (self.u == control_amps).all():
            self._time_evolution_energy(control_amps, update_idx)
            self._back_propagation_energy(control_amps, update_idx)
            # self.u = control_amps
            grad = np.zeros_like(control_amps)
            for k in range(self.n_ts):
                grad[k, 0] = -np.imag(self._onto[self.n_ts - k - 1].dot((-self.H_c[0]).dot(self._into[k + 1]))
                                      * self.delta_t)
                grad[k, 1] = -np.imag(self._onto[self.n_ts - k - 1].dot((-self.H_c[1]).dot(self._into[k + 1]))
                                      * self.delta_t)
            grad *= 2
            # grad = np.expand_dims(np.array(grad), 1)
        return grad

    def _compute_objective_value(self, control_amps, update_idx=-1):
        if self.obj_type == 'fid':
            obj = self.optim.fid_err_func_compute(control_amps.reshape(-1))
        if self.obj_type == 'energy':
            self._time_evolution_energy(control_amps, update_idx)
            obj = np.real(self._into[-1].conj().T.dot(self.H_c[1].dot(self._into[-1])))
        return obj

    def draw_extracted_control(self, fig_name):
        if self.obj_type == "energy":
            n_ctrl = 2
        else:
            n_ctrl = self.n_ctrl
        control = np.zeros((self.n_ts, n_ctrl))

        for t in range(self.n_ts):
            control[t, int(self.j_hat[t])] = 1

        t = np.linspace(0, self.evo_time, self.n_ts + 1)
        plt.figure(dpi=300)
        plt.xlabel("Time")
        plt.ylabel("Control amplitude")
        marker_list = ['-o', '--^', '-*', '--s']
        marker_size_list = [5, 5, 8, 5]
        marker_step = int(self.n_ts / 10)
        for j in range(n_ctrl):
            # if max(control[:, j]) > 0:
            plt.step(t, np.hstack((control[:, j], control[-1, j])), marker_list[j % 4],
                     where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, marker_step),
                     markersize=marker_size_list[j % 4])
        plt.legend()
        plt.savefig(fig_name)

    def draw_metric(self, fig_name, metric="gradient"):
        if self.obj_type == "energy":
            n_ctrl = 2
        else:
            n_ctrl = self.n_ctrl

        # draw the metric
        t = np.linspace(0, self.evo_time, self.n_ts + 1)
        plt.figure(dpi=300)
        plt.xlabel("Time")

        if metric == "sur":
            # initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
            # deviation = np.zeros((self.n_ts, n_ctrl))
            # deviation[0, :] = initial_control[0, :]
            # for k in range(1, self.n_ts):
            #     j_hat = np.argmax(deviation[k - 1, :])
            #     deviation[k, :] = deviation[k - 1, :] + initial_control[k, :]
            #     deviation[k, j_hat] = deviation[k - 1, j_hat] + initial_control[k, j_hat] - 1
            # metric_data = deviation
            metric_data = self.deviation
            plt.ylabel('deviation')
        if metric == "gradient":
            metric_data = -self.grad
            plt.ylabel('negative gradient')
        if metric == "la":
            metric_data = -self.la
            plt.ylabel('linear approximated change of objective value')
        if metric == "redu":
            metric_data = -self.cost
            plt.ylabel('change of objective value')
        if metric == "naive":
            initial_control = np.loadtxt(self.initial_control_file, delimiter=",")
            metric_data = initial_control
            plt.ylabel('Control value')

        # plt.ylabel("Metric")
        marker_list = ['-o', '--^', '-*', '--s']
        marker_size_list = [5, 5, 8, 5]
        marker_step = int(self.n_ts / 10)
        linestyle_list = ['-', '--', 'dashdot']
        for j in range(n_ctrl):
            # if max(control[:, j]) > 0:
            # plt.step(t, np.hstack((metric_data[:, j], metric_data[-1, j])), marker_list[j % 4],
            #          where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, marker_step),
            #          markersize=marker_size_list[j % 4])
            plt.step(t, np.hstack((metric_data[:, j], metric_data[-1, j])), linestyle=linestyle_list[j % 3],
                     where='post', linewidth=2, label='controller ' + str(j + 1))
        plt.legend()
        plt.savefig(fig_name)


if __name__ == '__main__':
    n = 4
    num_edges = 2
    seed = 1
    n_ts = 40
    evo_time = 2
    initial_type = "warm"
    alpha = 0.01
    min_up_time = 0
    name = "EnergySTTR"
    step = 1

    lb_threshold = 0.1
    ub_threshold = 0.65

    if seed == 0:
        Jij, edges = generate_Jij_MC(n, num_edges, 100)

    else:
        Jij = generate_Jij(n, seed)

    C = get_ham(n, True, Jij)
    B = get_ham(n, False, Jij)

    y0 = uniform(n)

    initial_control = "../example/control/ADMM/EnergyADMM4_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv"
    switches = Switches(initial_control, delta_t=evo_time / n_ts)
    warm_start_length, num_switch, ctrl_hamil_idx = switches.obtain_switches('naive')
    print(ctrl_hamil_idx)
    warm_start_length, num_switch, ctrl_hamil_idx = switches.obtain_switches('random', lb_threshold, ub_threshold)
    print(ctrl_hamil_idx)
    switches.init_gradient_computer(None, [B, C], y0[0:2 ** n], None, n_ts, evo_time, 'energy')
    warm_start_length, num_switch, ctrl_hamil_idx = switches.obtain_switches('gradient')
    print(ctrl_hamil_idx)
