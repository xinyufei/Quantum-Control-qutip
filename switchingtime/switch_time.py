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
from tools.evolution import compute_TV_norm
from tools.auxiliary_molecule import generate_molecule_func
from switchingtime.obtain_switches import Switches


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

        self.forward = None
        self.backward = None
        self.hamil_expm = None
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
        self.forward = [self.x0]
        self.hamil_expm = []
        for k in range(self.num_switch + 1):
            self.hamil_expm.append(expm(-1j * self.hlist[self.ctrl_hamil_idx[k]].copy() * x[k]))
            cur_state = self.hamil_expm[k].dot(self.forward[k])
            self.forward.append(cur_state)
        final_state = self.forward[-1]
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

    def gradient(self, x):
        if self.obj_type == 'energy':
            self.backward = [self.forward[-1].conj().T.dot(self.hlist[1].conj().T)]
        if self.obj_type == 'fid':
            self.backward = [self.xtarg.conj().T]
        for k in range(self.num_switch + 1):
            bwd = self.backward[k].dot(self.hamil_expm[self.num_switch - k])
            self.backward.append(bwd)
        grad = []
        if self.obj_type == 'energy':
            for k in range(self.num_switch + 1):
                grad += [np.imag(self.backward[self.num_switch - k].dot(
                    self.hlist[self.ctrl_hamil_idx[k]].copy().dot(self.forward[k + 1])))]
        if self.obj_type == 'fid':
            pre_grad = np.zeros(self.num_switch + 1, dtype=complex)
            for k in range(self.num_switch + 1):
                # grad_temp = expm_frechet(-1j * H[t] * delta_t, -1j * self.H_c[j] * delta_t, compute_expm=False)
                grad_temp = -1j * self.hlist[self.ctrl_hamil_idx[k]].copy()
                pre_grad[k] = np.trace(self.backward[self.num_switch - k].dot(grad_temp).dot(self.forward[k + 1]))
            fid_pre = np.trace(self.xtarg.conj().T.dot(self.forward[-1]))
            grad = - np.real(pre_grad * np.exp(-1j * np.angle(fid_pre)) / self.xtarg.shape[0])
        return grad

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
        bounds = Bounds([self.time_lb] * self.num_switch + [0], [self.time_ub] * (self.num_switch + 1))
        # minimize the objective function
        # res = minimize(self.objective, x0, method='SLSQP', constraints=eq_cons, bounds=bounds, options={'ftol': 1e-06})
        res = minimize(self.objective, x0, method='SLSQP', constraints=eq_cons, bounds=bounds,
                       jac=self.gradient, options={'ftol': 1e-06})
        self.tau = res.x
        # retrieve the switching time points
        self.retrieve_switching_points()
        # compute the final state and optimal value
        self.final_state = self.obtain_state(self.tau)
        self.obj = res.fun

        return res

    def optimize_minup(self):
        # predefine equality constraints for parameters
        num_var = int(len(self.tau0))
        eq_cons = {'type': 'eq',
                   'fun': lambda x: sum(x[0:num_var]) - self.evotime}
        cons = [eq_cons]
        cons.append({'type': 'eq',
                     'fun': lambda x: sum((x[num_var:] - 1/2)**2) - (num_var - 1)/4})
        cons.append({'type': 'ineq', 'fun': lambda x: x[0:num_var - 1] - self.time_lb * x[num_var:]})
        cons.append({'type': 'ineq', 'fun': lambda x: self.time_ub * x[num_var:] - x[0:num_var - 1]})
        # initialize the solution
        x0 = self.tau0.copy()
        # set the bounds of variables
        bounds = Bounds([0] * (2 * num_var - 1), [self.time_ub] * num_var + [1] * (num_var - 1))
        # minimize the objective function
        # res = minimize(self.objective, np.array(list(x0) + [0] * (num_var - 1)), method='SLSQP', constraints=cons,
        #                bounds=bounds, options={'ftol': 1e-06})
        # res = minimize(self.objective, np.array(list(x0) + list(np.random.rand(num_var - 1))), method='SLSQP', constraints=cons,
        #                bounds=bounds, options={'ftol': 1e-06})
        # init_aux = list(np.random.randint(2, size=num_var - 1))
        # print(init_aux)
        res = minimize(self.objective, np.array(list(x0) + init_aux), method='SLSQP',
                       constraints=cons,
                       bounds=bounds, options={'ftol': 1e-06})
        self.tau = res.x
        # retrieve the switching time points
        self.retrieve_switching_points()
        # compute the final state and optimal value
        self.final_state = self.obtain_state(self.tau)
        self.obj = res.fun

        return res

    def optimize_from_sur(self):
        """
        optimize the length of each control interval where the length each control interval can be zero
        :return: result of optimization
        """
        # predefine equality constraints for parameters
        eq_cons = {'type': 'eq',
                   'fun': lambda x: sum(x) - self.evotime}
        cons = [eq_cons]
        # for k in range(self.num_switch + 1):
        #     cons.append({'type': 'ineq', 'fun': lambda x: x[k] ** 2 - self.time_lb * x[k]})
        cons.append({'type': 'ineq', 'fun': lambda x: np.array([x[k] ** 2 - self.time_lb * x[k]
                                                                for k in range(self.num_switch + 1)])})

        # initialize the solution
        x0 = self.tau0.copy()
        # set the bounds of variables
        # bounds = Bounds([self.time_lb] * (self.num_switch + 1), [self.time_ub] * (self.num_switch + 1))
        bounds = Bounds([0] * (self.num_switch + 1), [self.time_ub] * (self.num_switch + 1))
        # minimize the objective function
        res = minimize(self.objective, x0, method='SLSQP', constraints=cons, bounds=bounds)
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
            for time_step in range(int(cur_time_l / delta_t), min(int(cur_time_r / delta_t), num_time_step)):
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


def obtain_switching_time(initial_control_file, delta_t=0.05):
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


def obtain_switching_time_from_sur(initial_control_file, delta_t=0.05):
    initial_control = np.loadtxt(initial_control_file, delimiter=",")
    num_switches = initial_control.shape[0] - 1
    hsequence = []
    j_hat = np.zeros(initial_control.shape[0])
    for k in range(initial_control.shape[0]):
        j_hat[k] = np.argmax(initial_control[k, :])
        hsequence.append(int(j_hat[k]))
    tau0 = [delta_t] * initial_control.shape[0]
    return tau0, num_switches, hsequence


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


def obtain_switching_time_uncertain(initial_control_file, delta_t=0.05, step=1, thre_max=0.65, thre_min=0.1, seed=None):
    # compute average initial control
    initial_control_pre = np.loadtxt(initial_control_file, delimiter=",")
    n_ts = initial_control_pre.shape[0]
    initial_control = np.zeros((int(np.ceil(n_ts / step)), initial_control_pre.shape[1]))
    for k in range(0, initial_control_pre.shape[0], step):
        initial_control[int(k / step), :] = sum(initial_control_pre[k + l, :] for l in range(step)) / step

    num_switches = 0
    hsequence = []
    tau0 = []
    j_hat = np.zeros(initial_control.shape[0])
    j_hat[0] = obtain_controller(initial_control[0, :], thre_max, thre_min, seed)
    hsequence.append(int(j_hat[0]))
    prev_switch = 0
    for k in range(1, initial_control.shape[0]):
        j_hat[k] = obtain_controller(initial_control[k, :], thre_max, thre_min, seed)
        if j_hat[k] != j_hat[k - 1]:
            num_switches += 1
            tau0.append(k * delta_t - prev_switch)
            prev_switch = k * delta_t
            hsequence.append(int(j_hat[k]))
    tau0.append(initial_control.shape[0] * delta_t - prev_switch)
    print(j_hat)
    return tau0, num_switches, hsequence


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
                                                      p=[1/len(min_interval)] * len(min_interval), replace=False))

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


def test_optimize():
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


def test_optimize_sur():
    n = 2
    num_edges = 1
    seed = 0
    n_ts = 40
    evo_time = 2
    initial_type = "warm"
    alpha = 0.01
    min_up_time = 0.5
    name = "EnergyST2SUR"

    Jij, edges = generate_Jij_MC(n, num_edges, 100)

    C = get_ham(n, True, Jij)
    B = get_ham(n, False, Jij)

    y0 = uniform(n)

    initial_control = "../example/control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_SUR.csv"
    warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time_from_sur(initial_control, evo_time / n_ts)

    # sequence of control hamiltonians
    ctrl_hamil = [B, C]

    # X_0 = np.expand_dims(y0[0:2**n], 1)
    X_0 = y0[0:2 ** n]

    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    energy_opt = SwitchTimeOpt()
    energy_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0, None, evo_time, num_switch, min_up_time, None,
        obj_type='energy')
    start = time.time()
    res = energy_opt.optimize_from_sur()
    end = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", energy_opt.switch_time, file=output_file)
    print("computational time", end - start, file=output_file)

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".csv"
    control = energy_opt.retrieve_control(n_ts)
    np.savetxt(control_name, control, delimiter=",")

    print("alpha", alpha, file=output_file)
    tv_norm = energy_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", energy_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".png"
    energy_opt.draw_control(figure_name)

    b_bin = np.loadtxt(control_name, delimiter=",")
    f = open(output_name, "a+")
    print("total tv norm", compute_TV_norm(b_bin), file=f)
    print("initial file", initial_control, file=f)
    f.close()


def test_optimize_sample():
    n = 4
    num_edges = 2
    seed = 5
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

    # initial_control = "../example/control/ADMM/EnergyADMM4_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv"
    initial_control = "../example/control/Trustregion/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    # initial_control = "../example/control/Continuous/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv"
    warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time_uncertain(
        initial_control, evo_time / n_ts, step, ub_threshold, lb_threshold)
    if min_up_time > 0:
        warm_start_length, num_switch, ctrl_hamil_idx = reduce_switch(
            evo_time / min_up_time - 1, warm_start_length, ctrl_hamil_idx)

    # sequence of control hamiltonians
    ctrl_hamil = [B, C]

    # X_0 = np.expand_dims(y0[0:2**n], 1)
    X_0 = y0[0:2 ** n]
        

    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    energy_opt = SwitchTimeOpt()
    energy_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0, None, evo_time, num_switch, min_up_time, None,
        obj_type='energy')
    start = time.time()
    res = energy_opt.optimize()
    end = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", energy_opt.switch_time, file=output_file)
    print("computational time", end - start, file=output_file)
    print("thresholds", lb_threshold, ub_threshold, file=output_file)

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".csv"
    control = energy_opt.retrieve_control(n_ts)
    np.savetxt(control_name, control, delimiter=",")

    print("alpha", alpha, file=output_file)
    tv_norm = energy_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", energy_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".png"
    energy_opt.draw_control(figure_name)

    b_bin = np.loadtxt(control_name, delimiter=",")
    f = open(output_name, "a+")
    print("total tv norm", compute_TV_norm(b_bin), file=f)
    print("initial file", initial_control, file=f)
    f.close()


def test_optimize_sample_compile():
    d = 2
    qubit_num = 4
    molecule = "LiH"
    target = "../example/control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv"
    # target = "../example/control/Continuous/MoleculeNEW_H2_evotime4.0_n_ts80_target.csv"
    initial_type = "warm"
    Hops, H0, U0, U = generate_molecule_func(qubit_num, d, molecule)
    
    if target is not None:
        U = np.loadtxt(target, dtype=np.complex_, delimiter=',')
    else:
        print("Please provide the target file!")
        exit()
        
    n_ts = 200
    evo_time = 20

    step = 1
    alpha = 0.001
    min_up_time = 0.5

    name = "MoleculeSTTR"

    # The control Hamiltonians (Qobj classes)
    H_c = [Qobj(hops) for hops in Hops]
    # Drift Hamiltonian
    H_d = Qobj(H0)
    # start point for the gate evolution
    X_0 = Qobj(U0)
    # Target for the gate evolution
    X_targ = Qobj(U)

    # initial_control = "../example/control/Continuous/MoleculeNEW_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv"
    # initial_control = "../example/control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv"
    # initial_control = "../example/control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv"
    # initial_control = "../example/control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100.csv"
    # initial_control = "../example/control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    initial_control = "../example/control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"

    if initial_control is None:
        print("Must provide control results of ADMM!")
        exit()

    lb_threshold = 0.1
    ub_threshold = 0.65

    warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time_uncertain(
        initial_control, evo_time / n_ts, step, thre_max=ub_threshold, thre_min=lb_threshold)

    if min_up_time > 0:
        warm_start_length, num_switch, ctrl_hamil_idx = reduce_switch(
            evo_time / min_up_time - 1, warm_start_length, ctrl_hamil_idx)

    print(num_switch)

    # sequence of control hamiltonians
    ctrl_hamil = [(H_d + H_c[j]).full() for j in range(len(H_c))]

    # initial control
    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    spin_opt = SwitchTimeOpt()
    min_time = 1
    spin_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0.full(), X_targ.full(), evo_time, num_switch, min_up_time,
        None)
    start = time.time()
    res = spin_opt.optimize()
    end = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", spin_opt.switch_time, file=output_file)
    print("computational time", end - start, file=output_file)
    print("thresholds", lb_threshold, ub_threshold, file=output_file)

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".csv"
    control = spin_opt.retrieve_control(n_ts)
    np.savetxt(control_name, control, delimiter=",")

    print("alpha", alpha, file=output_file)
    tv_norm = spin_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", spin_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".png"
    spin_opt.draw_control(figure_name)

    b_bin = np.loadtxt(control_name, delimiter=",")
    f = open(output_name, "a+")
    print("total tv norm", compute_TV_norm(b_bin), file=f)
    print("initial file", initial_control, file=f)
    f.close()


def test_optimize_minup():
    n = 4
    num_edges = 2
    seed = 5
    n_ts = 40
    evo_time = 2
    initial_type = "warm"
    alpha = 0.01
    min_up_time = 0.5
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

    # initial_control = "../example/control/ADMM/EnergyADMM4_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv"
    initial_control = "../example/control/Trustregion/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    # initial_control = "../example/control/Continuous/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv"
    warm_start_length, num_switch, ctrl_hamil_idx = obtain_switching_time_uncertain(
        initial_control, evo_time / n_ts, step, ub_threshold, lb_threshold)
    # if min_up_time > 0:
    #     warm_start_length, num_switch, ctrl_hamil_idx = reduce_switch(
    #         evo_time / min_up_time - 1, warm_start_length, ctrl_hamil_idx)

    # sequence of control hamiltonians
    ctrl_hamil = [B, C]

    # X_0 = np.expand_dims(y0[0:2**n], 1)
    X_0 = y0[0:2 ** n]

    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    energy_opt = SwitchTimeOpt()
    energy_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0, None, evo_time, num_switch, min_up_time, None,
        obj_type='energy')
    start = time.time()
    res = energy_opt.optimize_minup()
    end = time.time()
    print(res)
    print(end - start)


def test_optimize_gradient():
    n = 6
    num_edges = 3
    seed = 3
    n_ts = 40
    evo_time = 2
    initial_type = "warm"
    alpha = 0.01
    min_up_time = 0
    name = "EnergySTADMMG"

    lb_threshold = 0.1
    ub_threshold = 0.65

    if seed == 0:
        Jij, edges = generate_Jij_MC(n, num_edges, 100)

    else:
        Jij = generate_Jij(n, seed)

    C = get_ham(n, True, Jij)
    B = get_ham(n, False, Jij)

    y0 = uniform(n)

    initial_control = "../example/control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance3.csv"
    # initial_control = "../example/control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    # initial_control = "../example/control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv"
    switches = Switches(initial_control, delta_t=evo_time / n_ts)
    start1 = time.time()
    switches.init_gradient_computer(None, [B, C], y0[0:2 ** n], None, n_ts, evo_time, 'energy')
    warm_start_length, num_switch, ctrl_hamil_idx = switches.obtain_switches('gradient')
    end1 = time.time()
    print(ctrl_hamil_idx)
    if min_up_time > 0:
        warm_start_length, num_switch, ctrl_hamil_idx = reduce_switch(
            evo_time / min_up_time - 1, warm_start_length, ctrl_hamil_idx)

    # sequence of control hamiltonians
    ctrl_hamil = [B, C]

    # X_0 = np.expand_dims(y0[0:2**n], 1)
    X_0 = y0[0:2 ** n]

    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    energy_opt = SwitchTimeOpt()
    start2 = time.time()
    energy_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0, None, evo_time, num_switch, min_up_time, None,
        obj_type='energy')
    res = energy_opt.optimize()
    end2 = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", energy_opt.switch_time, file=output_file)
    print("computational time of retrieving switches", end1 - start1, file=output_file)
    print("computational time of optimization", end2 - start2, file=output_file)
    print("total computational time", end2 - start1, file=output_file)
    print("thresholds", lb_threshold, ub_threshold, file=output_file)

    exit()

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".csv"
    control = energy_opt.retrieve_control(n_ts)
    np.savetxt(control_name, control, delimiter=",")

    print("alpha", alpha, file=output_file)
    tv_norm = energy_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", energy_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}_instance{}".format(
        name + str(n), str(evo_time), str(n_ts), str(num_switch), initial_type, str(min_up_time), seed) + ".png"
    energy_opt.draw_control(figure_name)

    b_bin = np.loadtxt(control_name, delimiter=",")
    f = open(output_name, "a+")
    print("total tv norm", compute_TV_norm(b_bin), file=f)
    print("initial file", initial_control, file=f)
    f.close()


def test_optimize_gradient_compile():
    d = 2
    qubit_num = 4
    molecule = "LiH"
    target = "../example/control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv"
    # target = "../example/control/Continuous/MoleculeNEW_H2_evotime4.0_n_ts80_target.csv"
    initial_type = "warm"
    Hops, H0, U0, U = generate_molecule_func(qubit_num, d, molecule)

    if target is not None:
        U = np.loadtxt(target, dtype=np.complex_, delimiter=',')
    else:
        print("Please provide the target file!")
        exit()

    n_ts = 200
    evo_time = 20

    step = 1
    alpha = 0.001
    min_up_time = 0

    name = "MoleculeSTCG"

    # The control Hamiltonians (Qobj classes)
    H_c = [Qobj(hops) for hops in Hops]
    # Drift Hamiltonian
    H_d = Qobj(H0)
    # start point for the gate evolution
    X_0 = Qobj(U0)
    # Target for the gate evolution
    X_targ = Qobj(U)

    # initial_control = "../example/control/Continuous/MoleculeNEW_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv"
    initial_control = "../example/control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv"
    # initial_control = "../example/control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv"
    # initial_control = "../example/control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100.csv"
    # initial_control = "../example/control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"
    # initial_control = "../example/control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv"

    if initial_control is None:
        print("Must provide control results of ADMM!")
        exit()

    lb_threshold = 0.1
    ub_threshold = 0.65

    switches = Switches(initial_control, delta_t=evo_time / n_ts)
    start1 = time.time()
    switches.init_gradient_computer(H0, Hops, U0, U, n_ts, evo_time, 'fid')
    warm_start_length, num_switch, ctrl_hamil_idx = switches.obtain_switches('gradient')
    end1 = time.time()
    print(ctrl_hamil_idx)
    if min_up_time > 0:
        warm_start_length, num_switch, ctrl_hamil_idx = reduce_switch(
            evo_time / min_up_time - 1, warm_start_length, ctrl_hamil_idx)

    print(num_switch)

    # sequence of control hamiltonians
    ctrl_hamil = [(H_d + H_c[j]).full() for j in range(len(H_c))]

    # initial control
    if initial_type == "ave":
        initial = np.ones(num_switch + 1) * evo_time / (num_switch + 1)
    if initial_type == "rnd":
        initial_pre = np.random.random(num_switch + 1)
        initial = initial_pre.copy() / sum(initial_pre) * evo_time
    if initial_type == "warm":
        initial = warm_start_length

    # build optimizer
    spin_opt = SwitchTimeOpt()
    min_time = 1
    spin_opt.build_optimizer(
        ctrl_hamil, ctrl_hamil_idx, initial, X_0.full(), X_targ.full(), evo_time, num_switch, min_up_time,
        None)
    start2 = time.time()
    res = spin_opt.optimize()
    end2 = time.time()

    if not os.path.exists("../example/output/SwitchTime/test/"):
        os.makedirs("../example/output/SwitchTime/test/")
    if not os.path.exists("../example/control/SwitchTime/test/"):
        os.makedirs("../example/control/SwitchTime/test/")
    if not os.path.exists("../example/figure/SwitchTime/test/"):
        os.makedirs("../example/figure/SwitchTime/test/")

    # output file
    output_name = "../example/output/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".log"
    output_file = open(output_name, "a+")
    print(res, file=output_file)
    print("switching time points", spin_opt.switch_time, file=output_file)
    print("computational time of retrieving switches", end1 - start1, file=output_file)
    print("computational time of optimization", end2 - start2, file=output_file)
    print("total computational time", end2 - start1, file=output_file)
    print("thresholds", lb_threshold, ub_threshold, file=output_file)

    # retrieve control
    control_name = "../example/control/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".csv"
    control = spin_opt.retrieve_control(n_ts)
    np.savetxt(control_name, control, delimiter=",")

    print("alpha", alpha, file=output_file)
    tv_norm = spin_opt.tv_norm()
    print("tv norm", tv_norm, file=output_file)
    print("objective with tv norm", spin_opt.obj + alpha * tv_norm, file=output_file)

    # figure file
    figure_name = "../example/figure/SwitchTime/test/" + "{}_evotime_{}_n_ts{}_n_switch{}_init{}_minuptime{}".format(
        name + "_" + molecule, str(evo_time), str(n_ts), str(num_switch), initial_type,
        str(min_up_time)) + ".png"
    spin_opt.draw_control(figure_name)

    b_bin = np.loadtxt(control_name, delimiter=",")
    f = open(output_name, "a+")
    print("total tv norm", compute_TV_norm(b_bin), file=f)
    print("initial file", initial_control, file=f)
    f.close()


if __name__ == '__main__':
    test_optimize_gradient_compile()
