import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import Bounds
from scipy.optimize import minimize
from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

sys.path.append("..")
from tools.auxiliary_energy import *


class SwitchTimeOpt:
    """
    class for optimization with switching time points
    """
    def __init__(self):
        self.hlist = None  # list of all control Hamiltonians
        self.ctrl_hamil_idx = None  # control sequence of the Hamiltonians

        self.tau = None  # length of time for each interval (variables to be optimized)
        self.tau0 = None  # initialize the solution
        self.switch_time = None  # switching time points

        self.x0 = None  # initial state
        self.xtarg = None  # target state

        self.evotime = None  # total evolution time
        self.num_switch = None  # number of switches

        self.final_state = None  # final state
        self.obj = None  # optimal value

        self.control = None  # descritized control results

        self.time_lb = 0
        self.time_ub = self.evotime

        self.obj_type = None

    def build_optimizer(self, hlist, ctrl_hamil_idx, init, x0, xtarg, evotime, num_switch, time_lb=None, time_ub=None,
                        obj_type='fid'):
        self.hlist = hlist
        self.ctrl_hamil_idx = ctrl_hamil_idx
        self.tau0 = init
        self.x0 = x0
        self.xtarg = xtarg
        self.evotime = evotime
        self.num_switch = num_switch

        if time_lb:
            self.time_lb = time_lb
        if time_ub:
            self.time_ub = time_ub
        else:
            self.time_ub = self.evotime

        self.obj_type = obj_type

    def obtain_state(self, x):
        """
            conduct time evolution to obtain the state
            :param x: parameters beta and gamma
            :param hamil_idx: index list of Hamiltonians
            :return: final state
            """
        # conduct time evolution
        state = [self.x0]
        for k in range(self.num_switch + 1):
            cur_state = expm(-1j * self.hlist[self.ctrl_hamil_idx[k]].copy() * x[k]).dot(state[k])
            state.append(cur_state)
        final_state = state[-1]
        return final_state

    def objective(self, x):
        """
        compute the objective value
        :param x: parameters beta and gamma
        :param hamil_idx: index list of Hamiltonians
        :return: objective value
        """
        # conduct time evolution
        final_state = self.obtain_state(x)
        if self.obj_type == 'fid':
            fid = np.abs(np.trace(self.xtarg.conj().T.dot(final_state))) / self.xtarg.shape[0]
            return 1 - fid
        if self.obj_type == "energy":
            obj = np.real(final_state.conj().T.dot(self.hlist[1].dot(final_state)))[0]
            return obj

    def optimize(self):
        """
        optimize the length of each control interval
        :return: result of optimization
        """
        # predefine equality constraints for parameters
        eq_cons = {'type': 'eq',
                   'fun': lambda x: sum(x) - self.evotime}

        # initialize the solution
        x0 = self.tau0.copy()
        # set the bounds of variables
        bounds = Bounds([self.time_lb] * (self.num_switch + 1), [self.time_ub] * (self.num_switch + 1))
        # minimize the objective function
        res = minimize(self.objective, x0, method='SLSQP', constraints=eq_cons, bounds=bounds)
        self.tau = res.x
        # retrieve the switching time points
        self.retrieve_switching_points()
        # compute the final state and optimal value
        self.final_state = self.obtain_state(self.tau)
        self.obj = res.fun

        return res

    def retrieve_switching_points(self):
        """
        retrieve switching points from the length of control intervals
        """
        # retrieve switching time points
        if self.num_switch > 0:
            self.switch_time = np.zeros(self.num_switch)
            self.switch_time[0] = self.tau[0]
            for k in range(1, self.num_switch):
                self.switch_time[k] = self.switch_time[k - 1] + self.tau[k]

    def retrieve_control(self, num_time_step):
        """
        retrieve control results from solutions of QAOA
        :param num_time_step: number of time steps
        :return: control results
        """
        control = np.zeros((num_time_step, len(self.hlist)))
        delta_t = self.evotime / num_time_step

        cur_time_l = 0
        cur_time_r = 0
        for k in range(self.num_switch):
            cur_time_r = self.switch_time[k]
            for time_step in range(int(cur_time_l / delta_t), int(cur_time_r / delta_t)):
                control[time_step, self.ctrl_hamil_idx[k]] = 1
            cur_time_l = cur_time_r

        for time_step in range(int(cur_time_l / delta_t), num_time_step):
            control[time_step, self.ctrl_hamil_idx[self.num_switch]] = 1

        self.control = control

        return control

    def draw_control(self, fig_name):
        """
        draw control results
        """
        t = np.linspace(0, self.evotime, self.control.shape[0] + 1)
        plt.figure(dpi=300)
        # plt.title("Optimised Control Sequences")
        plt.xlabel("Time")
        plt.ylabel("Control amplitude")
        marker_list = ['-o', '--^', '-*', '--s']
        marker_size_list = [5, 5, 8, 5]
        if self.control.shape[1] == 1:
            plt.step(t, np.hstack((self.control[:, 0], self.control[-1, 0])),
                     where='post', linewidth=2, label='controller ' + str(1))
        else:
            for j in range(self.control.shape[1]):
                plt.step(t, np.hstack((self.control[:, j], self.control[-1, j])), marker_list[j % 4],
                         where='post', linewidth=2, label='controller ' + str(j+1), markevery=(j, 4),
                         markersize=marker_size_list[j % 4])
        plt.legend()
        plt.savefig(fig_name)

    def tv_norm(self):
        """
        compute the tv norm of control results
        :return: value of tv norm
        """
        return sum(sum(abs(self.control[tstep + 1, j] - self.control[tstep, j])
                       for tstep in range(self.control.shape[0] - 1))
                   for j in range(self.control.shape[1]))


def obtain_switching_time(initial_control_file, delta_t=0.05, threshold=0.15):
    initial_control = np.loadtxt(initial_control_file, delimiter=",")
    num_switches = 0

    # if initial_control.shape[1] == 2:
    #     tau0 = [0]
    #     time_point = [0]
    #     for k in range(initial_control.shape[0] - 1):
    #         if abs(initial_control[k, 0] - initial_control[k + 1, 0]) >= threshold:
    #             num_switches += 1
    #             tau0.append(k * delta_t - time_point[-1])
    #             time_point.append(k * delta_t)
    #     tau0.append(initial_control.shape[0] * delta_t - time_point[-1])
    #     tau0.pop(0)
    #
    #     return tau0, num_switches
    #
    # else:
    hsequence = []
    tau0 = []
    j_hat = np.zeros(initial_control.shape[0])
    j_hat[0] = np.argmax(initial_control[0, :])
    hsequence.append(int(j_hat[0]))
    prev_switch = 0
    for k in range(1, initial_control.shape[0]):
        j_hat[k] = np.argmax(initial_control[k, :])
        if j_hat[k] != j_hat[k - 1]:
            num_switches += 1
            tau0.append(k * delta_t - prev_switch)
            prev_switch = k * delta_t
            hsequence.append(int(j_hat[k]))
    tau0.append(initial_control.shape[0] * delta_t - prev_switch)
    return tau0, num_switches, hsequence



if __name__ == '__main__':
    # The control Hamiltonians (Qobj classes)
    H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
    # Drift Hamiltonian
    H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
    # start point for the gate evolution
    X_0 = identity(4)
    # Target for the gate evolution
    X_targ = cnot()

    # Time allowed for the evolution
    evo_time = 20
    # number of time steps
    time_step = evo_time * 20

    # number of switching points
    initial_control_file = "initial_admm/SPINWARMNEW2_evotime8_n_ts80_ptypeWARM_offset0_objUNIT_penalty0.001_sum_penalty0.01.csv"
    warm_start_length, num_switch = obtain_switching_time(initial_control_file)

    # sequence of control hamiltonians
    ctrl_hamil = [(H_d + H_c[0]).full(), (H_d + H_c[1]).full()]
    start = 1
    ctrl_hamil_idx = [(i + start) % 2 for i in range(num_switch + 1)]

    # initial control
    initial_type = "warm"
    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    cnot_opt = SwitchTimeOpt()
    cnot_opt.build_optimizer(ctrl_hamil, ctrl_hamil_idx, initial, X_0.full(), X_targ.full(), evo_time, num_switch)
    start = time.time()
    res = cnot_opt.optimize()
    end = time.time()

    # output file
    output_name = "output/CNOT_evotime" + "{}_n_switch{}_init{}_start{}".format(
        str(evo_time), str(num_switch), initial_type, str(ctrl_hamil_idx[0])) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", cnot_opt.switch_time, file=output_file)
    print("computational time", end - start, file=output_file)

    # retrieve control
    control_name = "control/CNOT_evotime" + "{}_n_switch{}_init{}_start{}_n_ts{}".format(
        str(evo_time), str(num_switch), initial_type, str(ctrl_hamil_idx[0]), str(time_step)) + ".csv"
    control = cnot_opt.retrieve_control(200)
    np.savetxt(control_name, control)

    alpha = 0.05
    print("alpha", alpha, file=output_file)
    tv_norm = cnot_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", cnot_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "figure/CNOT_evotime" + "{}_n_switch{}_init{}_start{}".format(
        str(evo_time), str(num_switch), initial_type, str(ctrl_hamil_idx[0])) + ".png"
    cnot_opt.draw_control(figure_name)

