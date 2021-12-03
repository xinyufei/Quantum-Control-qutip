import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import Bounds
from scipy.optimize import minimize
from qutip import identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
from qutip import Qobj

sys.path.append("..")
from tools.auxiliary_energy import *
from tools.evolution import compute_TV_norm, compute_obj_fid, time_evolution
from tools.auxiliary_molecule import generate_molecule_func
from switchingtime.obtain_switches import Switches


class SwitchtimePoint:
    def __init__(self, time=None, controller=None, ctrl_switch_idx=None):
        self.time = time
        self.controller = controller
        self.ctrl_switch_idx = ctrl_switch_idx
        # self.control_value = control_value


class SwitchTimeOptController:
    """
    class for optimization with switching time points
    """

    def __init__(self):
        self.hlist = None  # list of all control Hamiltonians
        self.num_controller = None
        self.ctrl_hamil_idx = None  # control sequence of the Hamiltonians
        self.h0 = None

        self.control_start = None

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
        self.penalty_para = 0.1
        self.penalty_para_initial = 0.1
        self.penalty = None
        self.obj_with_penalty = None
        self.penalty_max_ite = None
        self.penalty_threshold = None
        self.penalty_ite = None

    def build_optimizer(self, hlist, h0, control_start, init, x0, xtarg, evotime, num_switch,
                        time_lb=None, time_ub=None, obj_type='fid', penalty_para=0.1, max_ite=10,
                        threshold=1e-3):
        self.hlist = hlist
        self.num_controller = len(hlist)
        self.h0 = h0
        # self.ctrl_hamil_idx = ctrl_hamil_idx
        self.control_start = control_start
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
        self.penalty_para = penalty_para
        self.penalty_para_initial = penalty_para
        self.penalty_max_ite = max_ite
        self.penalty_threshold = threshold

    def hamiltonian_ctrl_val(self, j, t):
        cur_switch_time = self.switch_time[j]
        cur_controller_switch = -1
        pre_time = 0
        for time in cur_switch_time:
            if time > pre_time:
                cur_controller_switch += 1
            if t <= time:
                break
        cur_controller_switch = max(cur_controller_switch, 0)
        return (self.control_start[j] + cur_controller_switch) % 2

    def obtain_state(self, x):
        """
            conduct time evolution to obtain the state
            :param x: parameters beta and gamma
            :param hamil_idx: index list of Hamiltonians
            :return: final state
            """
        # extract all the switch time points
        # cur_idx = 0
        # self.switch_time = []
        # all_switch_time = []
        # for j in range(self.num_controller):
        #     cur_time = 0
        #     self.switch_time.append([])
        #     for k in range(self.num_switch[j] + 1):
        #         cur_time += x[cur_idx]
        #         all_switch_time.append(cur_time)
        #         self.switch_time[j].append(cur_time)
        #         cur_idx += 1
        # all_switch_time.sort()
        # # compute corresponding hamiltonian list
        # cur_hlist = []
        # for switch in all_switch_time:
        #     print(switch, [self.hamiltonian_ctrl_val(j, switch) for j in range(self.num_controller)])
        #     cur_hlist.append(sum(self.hamiltonian_ctrl_val(j, switch) * self.hlist[j]
        #                          for j in range(self.num_controller)) + self.h0)
        # all_switch_time.insert(0, 0)
        # # conduct time evolution
        # state = [self.x0]
        # for switch_idx in range(1, len(all_switch_time)):
        #     cur_state = expm(-1j * cur_hlist[switch_idx - 1].copy() * (
        #             all_switch_time[switch_idx] - all_switch_time[switch_idx - 1])).dot(state[switch_idx - 1])
        #     state.append(cur_state)
        # final_state = state[-1]


        # print("=========================================")
        cur_idx = 0
        self.switch_time = []
        all_switch_time = []
        for j in range(self.num_controller):
            cur_time = 0
            self.switch_time.append([])
            cur_controller_switch = 0
            for k in range(self.num_switch[j] + 1):
                cur_time += x[cur_idx]
                if x[cur_idx] > 1e-10:
                    cur_controller_switch += 1
                cur_switch_time = SwitchtimePoint(cur_time, j, cur_controller_switch)
                all_switch_time.append(cur_switch_time)
                self.switch_time[j].append(cur_time)
                cur_idx += 1
        # sort switching time points by ascending order
        all_switch_time.sort(key=lambda x: x.time, reverse=False)
        # compute corresponding hamiltonian list
        cur_hlist = []
        cur_ctrl_val = self.control_start.copy()
        for switch in all_switch_time:
            # print(switch.time, cur_ctrl_val)
            cur_hlist.append(sum(cur_ctrl_val[j] * self.hlist[j] for j in range(self.num_controller)) + self.h0)
            cur_controller = switch.controller
            cur_ctrl_val[cur_controller] = (self.control_start[cur_controller] + switch.ctrl_switch_idx) % 2

        all_switch_time.insert(0, SwitchtimePoint(0, 0, 0))
        state = [self.x0]
        for switch_idx in range(1, len(all_switch_time)):
            cur_state = expm(-1j * cur_hlist[switch_idx - 1].copy() * (
                    all_switch_time[switch_idx].time - all_switch_time[switch_idx - 1].time)).dot(state[switch_idx - 1])
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
            obj = np.real(final_state.conj().T.dot(self.hlist[1].dot(final_state)))
            return obj

    def objective_penalty(self, x):
        # conduct time evolution
        final_state = self.obtain_state(x)
        # penalty function
        self.penalty = 0
        cur_switch = 0
        for j in range(self.num_controller):
            # print(sum(x[cur_switch:self.num_switch[j] + 1 + cur_switch]))
            self.penalty += np.square(sum(x[cur_switch:self.num_switch[j] + 1 + cur_switch]) - self.evotime)
            cur_switch += self.num_switch[j] + 1

        if self.obj_type == 'fid':
            fid = np.abs(np.trace(self.xtarg.conj().T.dot(final_state))) / self.xtarg.shape[0]
            self.obj = 1 - fid
            return self.obj + self.penalty_para * self.penalty
        if self.obj_type == "energy":
            obj = np.real(final_state.conj().T.dot(self.hlist[1].dot(final_state)))
            self.obj = obj
            return obj + self.penalty_para * self.penalty

    def optimize_with_penalty(self):
        # initialize the solution
        x0 = self.tau0.copy()
        # set the bounds of variables
        bounds = Bounds([self.time_lb] * len(x0), [self.time_ub] * len(x0))
        # minimize the objective function
        res = minimize(self.objective_penalty, x0, bounds=bounds, options={'ftol': 1e-08, 'maxiter': 200})
        self.tau = res.x
        # retrieve the switching time points
        self.retrieve_switching_points()
        # compute the final state and optimal value
        self.final_state = self.obtain_state(self.tau)
        self.obj_with_penalty = res.fun

        return res

    def optimize_iterate(self):
        self.penalty_para = self.penalty_para_initial
        res = None
        for ite in range(self.penalty_max_ite):
            res = self.optimize_with_penalty()
            print(res)
            if self.penalty <= self.penalty_threshold:
                break
            self.penalty_para *= 10
            self.tau0 = res.x.copy()
        self.penalty_ite = ite + 1
        return res

    def optimize(self):
        """
        optimize the length of each control interval
        :return: result of optimization
        """
        # predefine equality constraints for parameters
        eq_cons = []
        # pre_idx = 0
        # cur_idx = 0
        num_vars = np.zeros(self.num_controller + 1, dtype=int)
        num_vars[0] = 0
        for j in range(self.num_controller):
            num_vars[j + 1] = int(num_vars[j] + self.num_switch[j] + 1)
        for j in range(self.num_controller):
            # cur_idx += (self.num_switch[j] + 1)
            cons = {'type': 'eq',
                    'fun': lambda x: sum(x[k] for k in range(num_vars[j], num_vars[j + 1])) - self.evotime}
            # pre_idx = cur_idx
            eq_cons.append(cons)
        # eq_cons = {'type': 'eq',
        #            'fun': lambda x: sum(x) - self.evotime}

        # initialize the solution
        x0 = self.tau0.copy()
        # set the bounds of variables
        bounds = Bounds([self.time_lb] * len(x0), [self.time_ub] * len(x0))
        # minimize the objective function
        res = minimize(self.objective, x0, constraints=eq_cons, bounds=bounds, options={'ftol': 1e-08, 'maxiter': 200})
        self.tau = res.x
        # retrieve the switching time points
        self.retrieve_switching_points()
        # compute the final state and optimal value
        self.final_state = self.obtain_state(self.tau)
        self.obj = res.fun

        return res

    def retrieve_switching_points(self):
        cur_idx = 0
        self.switch_time = []
        for j in range(self.num_controller):
            cur_time = 0
            self.switch_time.append([])
            for k in range(self.num_switch[j]):
                cur_time += self.tau[cur_idx]
                self.switch_time[j].append(cur_time)
                cur_idx += 1

    def retrieve_control(self, num_time_step):
        """
        retrieve control results from solutions of QAOA
        :param num_time_step: number of time steps
        :return: control results
        """
        control = np.zeros((num_time_step, len(self.hlist)))
        delta_t = self.evotime / num_time_step

        for j in range(self.num_controller):
            cur_time_l = 0
            cur_time_r = 0
            for k in range(self.num_switch[j]):
                cur_time_r = self.switch_time[j][k]
                for time_step in range(int(cur_time_l / delta_t), min(int(cur_time_r / delta_t), num_time_step)):
                    control[time_step, j] = (self.control_start[j] + k) % 2
                cur_time_l = cur_time_r

            for time_step in range(int(cur_time_l / delta_t), num_time_step):
                control[time_step, j] = (self.control_start[j] + self.num_switch[j]) % 2

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
                         where='post', linewidth=2, label='controller ' + str(j + 1), markevery=(j, 4),
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

    def recompute_objective(self):
        x = self.tau.copy()
        cur_switch = 0
        for j in range(self.num_controller):
            x[self.num_switch[j] + cur_switch] = self.evotime - sum(
                self.tau[cur_switch:self.num_switch[j] + cur_switch])
            cur_switch += self.num_switch[j] + 1
        return self.objective(x)


def reduce_switch(max_switch, interval_len, c_sequence, epsilon=1e-5):
    num_switch = len(c_sequence) - 1
    interval_len = np.array(interval_len)
    c_sequence = np.array(c_sequence)

    # interval_len = np.array([0.15, 0.45, 0.3, 0.05, 0.5, 0.05, 0.4, 0.1])
    # c_sequence = np.array([1,0,2,0,1,0,1,0])

    while num_switch > max_switch:
        min_interval = np.where(abs(interval_len - interval_len.min()) < epsilon)[0]
        dif_switch = int(num_switch - max_switch)
        f_min_interval = min_interval
        if dif_switch < len(min_interval):
            f_min_interval = np.sort(np.random.choice(min_interval, dif_switch,
                                                      p=[1 / len(min_interval)] * len(min_interval), replace=False))

        # delete the control
        for idx in range(len(f_min_interval)):
            k = f_min_interval[idx]
            if k - idx == 0:
                interval_len[k - idx + 1] += interval_len[k - idx]
            else:
                interval_len[k - idx - 1] += interval_len[k - idx]
            c_sequence = np.delete(c_sequence, k - idx)
            interval_len = np.delete(interval_len, k - idx)

        # merge the controls
        origin_num_switch = len(c_sequence)
        term_l = []
        for k in range(0, origin_num_switch):
            term_l.append(origin_num_switch)
            for l in range(k + 1, origin_num_switch):
                if c_sequence[l] == c_sequence[k]:
                    interval_len[k] += interval_len[l]
                else:
                    term_l[k] = l
                    break
        c_sequence = np.delete(c_sequence, [control for control in range(1, term_l[0])])
        interval_len = np.delete(interval_len, [control for control in range(1, term_l[0])])
        delete_ctrl = term_l[0] - 1
        for k in range(1, origin_num_switch):
            if term_l[k] != term_l[k - 1]:
                c_sequence = np.delete(c_sequence, [control for control in range(
                    k + 1 - delete_ctrl, term_l[k] - delete_ctrl)])
                interval_len = np.delete(interval_len, [control for control in range(
                    k + 1 - delete_ctrl, term_l[k] - delete_ctrl)])
                delete_ctrl += term_l[k] - k - 1

        num_switch = len(c_sequence) - 1

    return interval_len, num_switch, c_sequence


def test_optimize_cnot():
    n_ts = 100
    evo_time = 5
    initial_type = "ave"
    alpha = 0.01
    min_up_time = 0
    name = "CNOTST"
    penalty_para = 100
    error_threshold = 1e-4

    H_c = [tensor(sigmax(), identity(2)).full(), tensor(sigmay(), identity(2)).full()]
    # Drift Hamiltonian
    H_d = (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())).full()
    # start point for the gate evolution
    X_0 = identity(4).full()
    # Target for the gate evolution
    X_targ = cnot().full()

    lb_threshold = 0.5
    ub_threshold = 0.65

    initial_control = "../example/control/ADMM/CNOTADMM_evotime5.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv"
    # initial_control = "../example/control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    # initial_control = "../example/control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv"
    switches = Switches(initial_control, delta_t=evo_time / n_ts)
    start1 = time.time()
    switches.init_gradient_computer(H_d, H_c, X_0, X_targ, n_ts, evo_time, 'fid')
    initial, num_switch, control_start = switches.obtain_switches('multi', thre_min=lb_threshold)
    end1 = time.time()

    num_switch = [9, 9]
    initial = np.array([evo_time / (num_switch[0] + 1)] * (num_switch[0] + 1) +
                       [evo_time / (num_switch[1] + 1)] * (num_switch[1] + 1))

    # build optimizer
    cnot_opt = SwitchTimeOptController()
    start2 = time.time()
    cnot_opt.build_optimizer(
        H_c, H_d, control_start, np.array(initial), X_0, X_targ, evo_time, num_switch, min_up_time, None,
        obj_type='fid', penalty_para=penalty_para, threshold=error_threshold)
    res = cnot_opt.optimize_with_penalty()
    # res = cnot_opt.optimize_iterate()
    end2 = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_init{}_minuptime{}".format(
        name, str(evo_time), str(n_ts), initial_type, str(min_up_time)) + ".log"
    output_file = open(output_name, "w+")
    print(res, file=output_file)
    print("original objective function", cnot_opt.obj, file=output_file)
    print("penalty parameter", cnot_opt.penalty_para, file=output_file)
    print("error threshold", cnot_opt.penalty_threshold, file=output_file)
    print("penalty", cnot_opt.penalty, file=output_file)
    print("number of switches", num_switch, file=output_file)
    print("start control", control_start, file=output_file)
    print("switching time points", cnot_opt.switch_time, file=output_file)
    print("computational time of retrieving switches", end1 - start1, file=output_file)
    print("computational time of optimization", end2 - start2, file=output_file)
    print("objective value with equality constraint", cnot_opt.recompute_objective(), file=output_file)
    print("total computational time", end2 - start1, file=output_file)
    print("threshold", lb_threshold, file=output_file)

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_init{}_minuptime{}".format(
        name, str(evo_time), str(n_ts), initial_type, str(min_up_time)) + ".csv"
    control = cnot_opt.retrieve_control(100)
    np.savetxt(control_name, control)

    print("alpha", alpha, file=output_file)
    tv_norm = cnot_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", cnot_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_init{}_minuptime{}".format(
        name, str(evo_time), str(n_ts), initial_type, str(min_up_time)) + ".png"
    cnot_opt.draw_control(figure_name)
    output_file.close()


if __name__ == '__main__':
    test_optimize_cnot()
