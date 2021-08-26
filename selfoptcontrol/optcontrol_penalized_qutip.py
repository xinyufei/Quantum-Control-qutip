import time
import datetime
import numpy as np
from scipy.linalg import expm, expm_frechet
import scipy.optimize
import matplotlib.pyplot as plt
import qutip.control.pulseoptim as cpo
from qutip import Qobj


class Optcontrol_Penalized_Qutip():
    """
    optimal controller with ADMM algorithm to implement minimum up time constraint
    """

    def __init__(self):
        self.H_d = None
        self.H_c = None
        self.H_d_qobj = None
        self.H_c_qobj = None
        self.X_0 = None
        self.X_targ = None
        self.X_0_qobj = None
        self.X_targ_qobj = None
        self.n_ts = 0
        self.evo_time = None
        self.amp_lbound = None
        self.amp_ubound = None
        self.ops_max_amp = 1
        self.fid_err_targ = None
        self.min_grad = None
        self.max_iter_step = None
        self.max_wall_time_step = None
        self.obj_type = None
        self.phase_option = None
        self.p_type = None
        self.seed = None
        self.constant = None
        self.initial_control = None
        self.output_num = None
        self.output_fig = None
        self.output_control = None
        self.sum_cons_1 = False
        self.n_ctrls = None
        self.admm_err_targ = None
        self.time_optimize_start_step = 0
        self.num_iter_step = 0
        self.penalty = np.infty

        self.cur_obj = 0
        self.onto = [None] * (self.n_ts + 1)
        self.fwd = [None] * (self.n_ts + 1)
        self.max_controllers = 1

        self.u = None
        self.termination_reason = None

        self.result = None

    def build_optimizer(self, H_d, H_c, X_0, X_targ, n_ts, evo_time, amp_lbound=0, amp_ubound=1, ops_max_amp=1, 
                        fid_err_targ=1e-4, min_grad=1e-8, max_iter_step=500, max_wall_time_step=120,
                        fid_type="UNIT", phase_option="PSU", p_type="ZERO", seed=None, constant=0, initial_control=None,
                        output_num=None, output_fig=None, output_control=None, penalty=10, max_controllers=1):
        self.H_d_qobj = H_d
        self.H_c_qobj = H_c
        self.H_d = H_d.full()
        self.H_c = [h_c.full() for h_c in H_c]
        self.X_0_qobj = X_0
        self.X_targ_qobj = X_targ
        self.X_0 = X_0.full()
        self.X_targ = X_targ.full()
        self.n_ts = n_ts
        self.evo_time = evo_time
        self.amp_lbound = amp_lbound
        self.amp_ubound = amp_ubound
        self.ops_max_amp = ops_max_amp
        self.fid_err_targ = fid_err_targ
        self.min_grad = min_grad
        self.max_iter_step = max_iter_step
        self.max_wall_time_step = max_wall_time_step
        self.obj_type = fid_type
        self.phase_option = phase_option
        self.p_type = p_type
        self.constant = constant
        self.initial_control = initial_control
        self.output_num = output_num
        self.output_fig = output_fig
        self.output_control = output_control
        self.seed = seed
        self.penalty = penalty
        self.max_controllers = max_controllers

        # if self.sum_cons_1:
        #     H_c_origin = H_c
        #     # Controller Hamiltonian
        #     self.H_c = [H_c_origin[i] - H_c_origin[-1] for i in range(len(H_c_origin) - 1)]
        #     # Drift Hamiltonian
        #     self.H_d = H_d + H_c_origin[-1]

        self.n_ctrls = len(self.H_c)
        if not isinstance(ops_max_amp, list):
            self.ops_max_amp = [ops_max_amp] * self.n_ctrls
        self.u = np.zeros((self.n_ts, self.n_ctrls))

        self.onto = [None] * (self.n_ts + 1)
        self.fwd = [None] * (self.n_ts + 1)

    def _initialize_control(self):
        """
        :param self:
        :return: an n_ts*n_ctrls array
        """
        self.init_amps = np.zeros([self.n_ts, self.n_ctrls])
        if self.p_type == "RND":
            if self.seed:
                np.random.seed(self.seed)
            self.init_amps = np.random.rand(
                self.n_ts, self.n_ctrls) * (self.amp_ubound - self.amp_lbound) + self.amp_lbound
        if self.p_type == "CONSTANT":
            self.init_amps = np.zeros((self.n_ts, self.n_ctrls)) + self.constant
        if self.p_type == "WARM":
            # file = open(self.initial_control)
            warm_start_control = np.loadtxt(self.initial_control, delimiter=",")
            evo_time_start = warm_start_control.shape[0]
            step = self.n_ts / evo_time_start
            for j in range(self.n_ctrls):
                for time_step in range(self.n_ts):
                    self.init_amps[time_step, j] = warm_start_control[int(np.floor(time_step / step)), j]

    def evolution(self, control_amps):
        delta_t = self.evo_time / self.n_ts
        X = [self.X_0]
        for t in range(self.n_ts):
            H_t = self.H_d.copy()
            for j in range(self.n_ctrls):
                H_t += control_amps[t, j] * self.ops_max_amp[j] * self.H_c[j].copy()
            X_t = expm(-1j * H_t * delta_t).dot(X[t])
            X.append(X_t)
        self.fwd = X
        return X[-1]

    def compute_fid(self, evolution_result):
        fid = 0
        if self.obj_type == "UNIT" and self.phase_option == "PSU":
            fid = np.abs(np.trace(
                np.linalg.inv(self.X_targ).dot(evolution_result))) / self.X_targ.shape[0]
        return fid

    def compute_penalty(self, control_amps):
        penalized = self.penalty * sum(np.power(sum(control_amps[t, j] for j in range(self.n_ctrls))
                                                - self.max_controllers, 2) for t in range(self.n_ts))
        return penalized

    def _compute_err(self, *args):
        """
        :param args: control list
        :return: error
        """
        control_amps = args[0].copy()
        control_amps = control_amps.reshape([self.n_ts, self.n_ctrls])
        evolution_result = self.evolution(control_amps)
        fid = self.compute_fid(evolution_result)
        penalized = self.compute_penalty(control_amps)
        # print(1 - fid)
        return 1 - fid + penalized
        # return 1 - fid

    def _step_call_back(self, *args):
        wall_time_step = time.time() - self.time_optimize_start_step
        # if wall_time_step > self.max_wall_time_step:
        #     raise ValueError("The time exceeds the given max wall time.")
        self.num_iter_step += 1

    def _penalized_gradient(self, control_amps):
        penalized_grad = np.zeros((self.n_ts, self.n_ctrls))
        for t in range(self.n_ts):
            grad_p = 2 * self.penalty * (sum(control_amps[t, j] for j in range(self.n_ctrls)) - self.max_controllers)
            for j in range(self.n_ctrls):
                penalized_grad[t, j] = grad_p
        return penalized_grad

    def _fprime(self, *args):
        control_amps = args[0].copy().reshape([self.n_ts, self.n_ctrls])
        delta_t = self.evo_time / self.n_ts
        fwd = [self.X_0]
        onto = [self.X_targ.conj().T]
        H = [None] * self.n_ts
        for t in range(self.n_ts):
            H[t] = self.H_d.copy()
            for j in range(self.n_ctrls):
                H[t] += control_amps[t, j] * self.ops_max_amp[j] * self.H_c[j].copy()
            cur_fwd = expm(-1j * H[t] * delta_t).dot(fwd[-1])
            fwd.append(cur_fwd)

            H_t_onto = self.H_d.copy()
            for j in range(self.n_ctrls):
                H_t_onto += control_amps[self.n_ts - t - 1, j] * self.ops_max_amp[j] * self.H_c[j].copy()
            cur_onto = onto[0].dot(expm(-1j * H_t_onto * delta_t))
            onto.insert(0, cur_onto)

        onto = np.array(onto)
        fwd = np.array(fwd)
        grad = np.zeros((self.n_ts, self.n_ctrls), dtype=complex)
        for t in range(self.n_ts):
            for j in range(self.n_ctrls):
                grad_temp = expm_frechet(-1j * H[t] * delta_t, -1j * self.ops_max_amp[j] * self.H_c[j] * delta_t, compute_expm=False)
                g = np.trace(onto[t + 1].dot(grad_temp).dot(fwd[t]))
                grad[t, j] = g
        fid_pre = np.trace(self.X_targ.conj().T.dot(fwd[-1]))
        fid_grad = - np.real(grad * np.exp(-1j * np.angle(fid_pre)) / self.X_targ.shape[0]).flatten()

        penalized_grad = self._penalized_gradient(control_amps)

        return fid_grad + penalized_grad.flatten()

    def _minimize_u(self):
        self.time_optimize_start_step = time.time()
        self.num_iter_step = 0
        # results = scipy.optimize.minimize(self._compute_err, self.init_amps.reshape(-1), method='L-BFGS-B',
        #                                   bounds=scipy.optimize.Bounds(self.amp_lbound, self.amp_ubound),
        #                                   tol=self.min_grad,
        #                                   options={"maxiter": self.max_iter_step}, callback=self._step_call_back)
        # initial_grad = self._fprime(self.u.reshape(-1))
        # threshold = 1e-2
        # min_grad = max(np.linalg.norm(initial_grad) * threshold, self.min_grad)
        min_grad = self.min_grad
        # f = open(self.output_num, "a+")
        # print(min_grad, file=f)
        optim = cpo.create_pulse_optimizer(self.H_d_qobj,
                                           [Qobj(self.H_c[j] * self.ops_max_amp[j]) for j in range(len(self.H_c))],
                                           self.X_0_qobj, self.X_targ_qobj, self.n_ts, self.evo_time,
                                           amp_lbound=self.amp_lbound, amp_ubound=self.amp_ubound,
                                           fid_err_targ=self.fid_err_targ, min_grad=min_grad,
                                           max_iter=self.max_iter_step, max_wall_time=self.max_wall_time_step,
                                           dyn_type='UNIT',
                                           fid_type=self.obj_type, phase_option="PSU",
                                           init_pulse_params={"offset": self.constant},
                                           gen_stats=True)
        optim.sum_penalty = self.penalty
        optim.max_controller = self.max_controllers
        dyn = optim.dynamics
        dyn.initialize_controls(self.init_amps)
        result = optim.run_optimization_sum_penalty()
        self.u = result.final_amps
        self.cur_obj = result.fid_err + self.compute_penalty(self.u)
        self.cur_grad = result.grad_norm_final
        self.num_iter_step = result.num_iter
        self.termination_reason = result.termination_reason
        self.result = result
        # results = scipy.optimize.fmin_l_bfgs_b(self._compute_err, self.init_amps.reshape(-1),
        #                                        bounds=[(self.amp_lbound, self.amp_ubound)] * self.n_ts * self.n_ctrls,
        #                                        pgtol=min_grad, fprime=self._fprime,
        #                                        maxiter=self.max_iter_step, callback=self._step_call_back)
        # self.u = results[0].reshape((self.n_ts, self.n_ctrls)).copy()
        # self.cur_obj = results[1]
        # self.cur_grad = results[2]['grad']
        # self.u = results.x.reshape((self.n_ts, self.n_ctrls)).copy()
        # self.cur_obj = results.fun

    def optimize_penalized(self):
        self._initialize_control()
        initial_amps = self.init_amps.copy()
        start = time.time()
        self._minimize_u()
        end = time.time()

        # output the results
        evo_full_final = self.evolution(self.u)
        fid = self.compute_fid(evo_full_final)
        report = open(self.output_num, "w+")
        print("Final evolution\n{}\n".format(evo_full_final), file=report)
        print("********* Summary *****************", file=report)
        print("Final fidelity error {}".format(1 - fid), file=report)
        print("Final objective value {}".format(self.cur_obj), file=report)
        print("Final penalized error {}".format(self.compute_penalty(self.u)), file=report)
        print("Final gradient {}".format(self.cur_grad), file=report)
        print("Number of iterations {}".format(self.num_iter_step), file=report)
        print("Terminated due to {}".format(self.termination_reason), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=end - start)), file=report)
        print("Computational time {}".format(end - start), file=report)
        self.result.stats.report()

        # output the control
        final_amps = np.zeros((self.n_ts, self.n_ctrls))
        for j in range(self.n_ctrls):
            final_amps[:, j] = self.u[:, j]

        if self.output_control:
            np.savetxt(self.output_control, final_amps, delimiter=",")

        # output the figures
        time_list = np.array([t * self.evo_time / self.n_ts for t in range(self.n_ts + 1)])
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.set_title("Initial control amps")
        # ax1.set_xlabel("Time")
        ax1.set_ylabel("Control amplitude")
        for j in range(self.n_ctrls):
            ax1.step(time_list, np.hstack((initial_amps[:, j], initial_amps[-1, j])), where='post')

        ax2 = fig1.add_subplot(2, 1, 2)
        ax2.set_title("Optimised Control Sequences")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Control amplitude")
        for j in range(final_amps.shape[1]):
            ax2.step(time_list, np.hstack((final_amps[:, j], final_amps[-1, j])), where='post')
        # if self.sum_cons_1:
        #     ax2.step(np.array([t for t in range(self.n_ts)]),
        #              np.hstack((final_amps[:, self.n_ctrls], final_amps[-1, self.n_ctrls])), where='post')
        plt.tight_layout()
        if self.output_fig:
            plt.savefig(self.output_fig)
