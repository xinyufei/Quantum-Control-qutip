import time
import datetime
import numpy as np
from scipy.linalg import expm, expm_frechet
import qutip.control.pulseoptim as cpo
import scipy.optimize
import matplotlib.pyplot as plt


class Optcontrol_ADMM_CNOT():
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

        self.cur_obj = 0
        self.cur_origin_obj = 0
        self.onto = [None] * (self.n_ts + 1)
        self.fwd = [None] * (self.n_ts + 1)

        # variables and parameters for ADMM
        self.v = None
        self.u = None
        self._lambda = None
        self.rho = None
        self.alpha = None
        self.err_list = []
        self.obj_list = []
        self.max_iter_admm = None
        self.max_wall_time_admm = None
        self.result = None
        self.qutip_optimizer = None

    def build_optimizer(self, H_d, H_c, X_0, X_targ, n_ts, evo_time, amp_lbound=0, amp_ubound=1,
                        fid_err_targ=1e-4, min_grad=1e-8, max_iter_step=500, max_wall_time_step=120,
                        fid_type="UNIT", phase_option="PSU", p_type="ZERO", seed=None, constant=0, initial_control=None,
                        output_num=None, output_fig=None, output_control=None, sum_cons_1=False,
                        alpha=1, rho=2, max_iter_admm=500, max_wall_time_admm=7200, admm_err_targ=1e-3):
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
        self.sum_cons_1 = sum_cons_1
        self.max_iter_admm = max_iter_admm
        self.max_wall_time_admm = max_wall_time_admm
        self.admm_err_targ = admm_err_targ
        self.alpha = alpha * 2
        self.rho = rho
        self.seed = seed

        if self.sum_cons_1:
            H_c_origin = H_c
            # Controller Hamiltonian
            self.H_c = [H_c_origin[i].full() - H_c_origin[-1].full() for i in range(len(H_c_origin) - 1)]
            self.H_c_qobj = [H_c_origin[i] - H_c_origin[-1] for i in range(len(H_c_origin) - 1)]
            # Drift Hamiltonian
            self.H_d = H_d.full() + H_c_origin[-1].full()
            self.H_d_qobj = H_d + H_c_origin[-1]

        self.n_ctrls = len(self.H_c)
        self.rho = rho
        self.u = np.zeros((self.n_ts, self.n_ctrls))
        if self.sum_cons_1:
            self.v = np.zeros((self.n_ts - 1, self.n_ctrls + 1))
            self._lambda = np.zeros((self.n_ts - 1, self.n_ctrls + 1))
        else:
            self.v = np.zeros((self.n_ts - 1, self.n_ctrls))
            self._lambda = np.zeros((self.n_ts - 1, self.n_ctrls))

        self.cur_obj = 0
        self.onto = [None] * (self.n_ts + 1)
        self.fwd = [None] * (self.n_ts + 1)

        optim = cpo.create_pulse_optimizer(self.H_d_qobj, self.H_c_qobj, self.X_0_qobj, self.X_targ_qobj,
                                           self.n_ts, self.evo_time,
                                           amp_lbound=self.amp_lbound, amp_ubound=self.amp_ubound,
                                           fid_err_targ=self.fid_err_targ, min_grad=min_grad,
                                           max_iter=self.max_iter_step, max_wall_time=self.max_wall_time_step,
                                           dyn_type='UNIT',
                                           fid_type=self.obj_type, phase_option="PSU",
                                           init_pulse_params={"offset": self.constant},
                                           gen_stats=True)
        self.qutip_optimizer = optim

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
        if self.p_type == "ADMM":
            self.init_amps = self.u.copy()

    def evolution(self, control_amps):
        delta_t = self.evo_time / self.n_ts
        X = [self.X_0]
        for t in range(self.n_ts):
            H_t = self.H_d.copy()
            for j in range(self.n_ctrls):
                H_t += control_amps[t, j] * self.H_c[j].copy()
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

    def compute_norm(self, control_amps):
        norm = sum(sum(np.power(control_amps[time_step + 1, j] - control_amps[time_step, j] - self.v[time_step, j]
                                + self._lambda[time_step, j], 2) for time_step in range(self.n_ts - 1))
                   for j in range(self.n_ctrls))
        if self.sum_cons_1:
            norm += sum(np.power(sum(control_amps[time_step, j] - control_amps[time_step + 1, j]
                                     for j in range(self.n_ctrls)) - self.v[time_step, self.n_ctrls]
                                 + self._lambda[time_step, self.n_ctrls], 2) for time_step in range(self.n_ts - 1))
        return norm

    def compute_tv_norm(self):
        return sum(sum(abs(self.u[t + 1, j] - self.u[t, j]) for t in range(self.n_ts - 1)) for j in range(self.n_ctrls))

    def _compute_err(self, *args):
        """
        :param args: control list
        :return: error
        """
        control_amps = args[0].copy()
        control_amps = control_amps.reshape([self.n_ts, self.n_ctrls])
        evolution_result = self.evolution(control_amps)
        fid = self.compute_fid(evolution_result)
        # norm = sum(sum(np.power(control_amps[time_step + 1, j] - control_amps[time_step, j] - self.v[time_step, j]
        #                         + self._lambda[time_step, j], 2) for time_step in range(self.n_ts - 1))
        #            for j in range(self.n_ctrls))
        norm = self.compute_norm(control_amps)
        # print(1 - fid)
        return 1 - fid + self.rho / 2 * norm
        # return 1 - fid

    def _step_call_back(self, *args):
        wall_time_step = time.time() - self.time_optimize_start_step
        # if wall_time_step > self.max_wall_time_step:
        #     raise ValueError("The time exceeds the given max wall time.")
        self.num_iter_step += 1

    def _fprime(self, *args):
        control_amps = args[0].copy().reshape([self.n_ts, self.n_ctrls])
        delta_t = self.evo_time / self.n_ts
        fwd = [self.X_0]
        onto = [self.X_targ.conj().T]
        H = [None] * self.n_ts
        for t in range(self.n_ts):
            H[t] = self.H_d.copy()
            for j in range(self.n_ctrls):
                H[t] += control_amps[t, j] * self.H_c[j].copy()
            cur_fwd = expm(-1j * H[t] * delta_t).dot(fwd[-1])
            fwd.append(cur_fwd)

            H_t_onto = self.H_d.copy()
            for j in range(self.n_ctrls):
                H_t_onto += control_amps[self.n_ts - t - 1, j] * self.H_c[j].copy()
            cur_onto = onto[0].dot(expm(-1j * H_t_onto * delta_t))
            onto.insert(0, cur_onto)

        onto = np.array(onto)
        fwd = np.array(fwd)
        grad = np.zeros((self.n_ts, self.n_ctrls), dtype=complex)
        for t in range(self.n_ts):
            for j in range(self.n_ctrls):
                grad_temp = expm_frechet(-1j * H[t] * delta_t, -1j * self.H_c[j] * delta_t, compute_expm=False)
                g = np.trace(onto[t + 1].dot(grad_temp).dot(fwd[t]))
                grad[t, j] = g
        fid_pre = np.trace(self.X_targ.conj().T.dot(fwd[-1]))
        fid_grad = - np.real(grad * np.exp(-1j * np.angle(fid_pre)) / self.X_targ.shape[0]).flatten()

        norm_grad = np.zeros((self.n_ts, self.n_ctrls))
        for j in range(self.n_ctrls):
            norm_grad[0, j] = -self.rho * (control_amps[1, j] - control_amps[0, j] - self.v[0, j] + self._lambda[0, j])\
                              + self.rho * (sum(control_amps[0, j] - control_amps[1, j] for j in range(self.n_ctrls))
                                            - self.v[0, self.n_ctrls] + self._lambda[0, self.n_ctrls])
            norm_grad[self.n_ts - 1, j] = self.rho * (control_amps[self.n_ts - 1, j] - control_amps[self.n_ts - 2, j]
                                                      - self.v[self.n_ts - 2, j] + self._lambda[self.n_ts - 2, j]) \
                                          - self.rho * (sum(
                control_amps[self.n_ts - 2, j] - control_amps[self.n_ts - 1, j] for j in range(self.n_ctrls))
                                                        - self.v[self.n_ts - 2, self.n_ctrls] + self._lambda[
                                                            self.n_ts - 2, self.n_ctrls])
            for t in range(1, self.n_ts - 1):
                norm_grad[t, j] = self.rho * (control_amps[t, j] - control_amps[t - 1, j] - self.v[t - 1, j]
                                              + self._lambda[t - 1, j]) \
                                  - self.rho * (control_amps[t + 1, j] - control_amps[t, j] - self.v[t, j]
                                                + self._lambda[t, j]) \
                                  + self.rho * (sum(control_amps[t, j] - control_amps[t + 1, j]
                                                    for j in range(self.n_ctrls))
                                                - self.v[t, self.n_ctrls] + self._lambda[t, self.n_ctrls]) \
                                  - self.rho * (sum(control_amps[t - 1, j] - control_amps[t, j]
                                                    for j in range(self.n_ctrls))
                                                - self.v[t - 1, self.n_ctrls] + self._lambda[t - 1, self.n_ctrls])
        return fid_grad + norm_grad.flatten()

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
        # results = scipy.optimize.fmin_l_bfgs_b(self._compute_err, self.init_amps.reshape(-1),
        #                                        bounds=[(self.amp_lbound, self.amp_ubound)] * self.n_ts * self.n_ctrls,
        #                                        pgtol=min_grad, fprime=self._fprime,
        #                                        maxiter=self.max_iter_step, callback=self._step_call_back)
        # self.u = results[0].reshape((self.n_ts, self.n_ctrls)).copy()
        # self.cur_obj = results[1]
        self.qutip_optimizer.termination_conditions.min_gradient_norm = min_grad
        self.qutip_optimizer.ADMM_rho = self.rho
        self.qutip_optimizer.v = self.v.copy()
        self.qutip_optimizer._lambda = self._lambda.copy()
        dyn = self.qutip_optimizer.dynamics
        dyn.initialize_controls(self.init_amps)
        result = self.qutip_optimizer.run_optimization_tv_penalty()
        self.u = result.final_amps
        self.cur_obj = result.fid_err + self.rho / 2 * self.compute_norm(self.u)
        self.cur_origin_obj = result.fid_err + self.alpha * self.compute_tv_norm()
        # self.cur_grad = result.grad_norm_final
        self.num_iter_step = result.num_iter
        # self.termination_reason = result.termination_reason
        self.result = result

    def _minimize_v(self):
        for j in range(self.n_ctrls):
            for t in range(self.n_ts - 1):
                temp = self.u[t + 1, j] - self.u[t, j] + self._lambda[t, j]
                if temp > self.alpha / self.rho:
                    self.v[t, j] = -self.alpha / self.rho + temp
                if temp < -self.alpha / self.rho:
                    self.v[t, j] = self.alpha / self.rho + temp
                if -self.alpha / self.rho <= temp <= self.alpha / self.rho:
                    self.v[t, j] = 0
        if self.sum_cons_1:
            for t in range(self.n_ts - 1):
                temp = sum(self.u[t, j] - self.u[t + 1, j] for j in range(self.n_ctrls)) + self._lambda[t, self.n_ctrls]
                if temp > self.alpha / self.rho:
                    self.v[t, self.n_ctrls] = -self.alpha / self.rho + temp
                if temp < -self.alpha / self.rho:
                    self.v[t, self.n_ctrls] = self.alpha / self.rho + temp
                if -self.alpha / self.rho <= temp <= self.alpha / self.rho:
                    self.v[t, self.n_ctrls] = 0

    def _update_dual(self):
        for j in range(self.n_ctrls):
            for t in range(self.n_ts - 1):
                self._lambda[t, j] += self.u[t + 1, j] - self.u[t, j] - self.v[t, j]
        if self.sum_cons_1:
            for t in range(self.n_ts - 1):
                self._lambda[t, self.n_ctrls] += sum(self.u[t, j] - self.u[t + 1, j] for j in range(self.n_ctrls))\
                                                 - self.v[t, self.n_ctrls]

    def _admm_err(self):
        err = sum(sum(np.power(self.u[t + 1, j] - self.u[t, j] - self.v[t, j], 2) for j in range(self.n_ctrls))
                  for t in range(self.n_ts - 1))
        if self.sum_cons_1:
            err += sum(np.power(sum(self.u[t, j] - self.u[t + 1, j] for j in range(self.n_ctrls))
                                - self.v[t, self.n_ctrls], 2) for t in range(self.n_ts - 1))
        return err

    def optimize_admm(self):
        self._initialize_control()
        initial_amps = self.init_amps.copy()
        self.v = np.zeros((self.n_ts - 1, self.n_ctrls))
        self._lambda = np.zeros((self.n_ts - 1, self.n_ctrls))
        if self.sum_cons_1:
            self.v = np.zeros((self.n_ts - 1, self.n_ctrls + 1))
            self._lambda = np.zeros((self.n_ts - 1, self.n_ctrls + 1))
        self.p_type = "ADMM"
        admm_start = time.time()
        self.admm_num_iter = 0
        time_iter = [0]
        threshold = 1e-1
        while 1:
            self.admm_num_iter += 1
            if self.admm_num_iter > 1:
                self._initialize_control()
            self._minimize_u()
            self._minimize_v()
            self._update_dual()
            err = self._admm_err()
            # if admm_num_iter == 1:
            #     err_0 = err
            self.err_list.append(err)
            self.obj_list.append(self.cur_obj)
            # norm = self.compute_norm(self.u)
            admm_opt_time = time.time()
            # self.admm_err_targ = threshold * err_0
            # if err < self.admm_err_targ:
            #     tr = "Achieve the error target of ADMM"
            #     break
            # time_iter.append(admm_opt_time - admm_start)
            time_iteration = admm_opt_time - admm_start - time_iter[-1]
            time_iter.append(time_iteration)
            if admm_opt_time - admm_start >= self.max_wall_time_admm:
                tr = "Exceed the max wall time of ADMM"
                break
            if self.admm_num_iter >= self.max_iter_admm:
                tr = "Exceed the maximum number of iteration of ADMM"
                break

        # output the results
        evo_full_final = self.evolution(self.u)
        fid = self.compute_fid(evo_full_final)
        report = open(self.output_num, "a+")
        print("Final evolution\n{}\n".format(evo_full_final), file=report)
        print("********* Summary *****************", file=report)
        print("Final fidelity error {}".format(1 - fid), file=report)
        print("Final objective value {}".format(self.cur_origin_obj), file=report)
        print("Final penalized TV regularizer {}".format(self.alpha * self.compute_tv_norm()), file=report)
        print("Final norm value {}".format(2 * self.compute_tv_norm()), file=report)
        print("Final error {}".format(self.err_list[-1]), file=report)
        print("Terminate reason {}".format(tr), file=report)
        print("Number of iterations {}".format(self.admm_num_iter), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=admm_opt_time - admm_start)), file=report)
        print("Computational time {}".format(admm_opt_time - admm_start), file=report)
        print("Time for each iteration", time_iter[1:], file=report)
    
        # output the control
        # final_amps = np.zeros((self.n_ts, self.n_ctrls))
        # if self.sum_cons_1:
        #     final_amps = np.zeros((self.n_ts, self.n_ctrls + 1))
        # for j in range(self.n_ctrls):
        #     final_amps[:, j] = self.u[:, j]
        # if self.sum_cons_1:
        final_amps = np.zeros((self.n_ts, 2))
        for t in range(self.n_ts):
            final_amps[t, 1] = 1 - self.u[t, 0]
            final_amps[t, 0] = self.u[t, 0]

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
        if self.n_ctrls == 1:
            ax1.step(time_list, np.hstack((1 - initial_amps[:, 0], 1 - initial_amps[-1, 0])), where='post')
        if self.sum_cons_1:
            ax1.step(time_list, np.hstack((1 - sum(initial_amps[:, j] for j in range(self.n_ctrls)),
                                           1 - sum(initial_amps[-1, j] for j in range(self.n_ctrls)))), where='post')

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

        # output the objective value figure and error figure
        plt.figure(dpi=300)
        # plt.title("Objective value")
        plt.xlabel("Iterations")
        plt.ylabel("Objective value")
        plt.plot(self.obj_list, label="objective value")
        plt.legend()
        plt.savefig(self.output_fig.split(".png")[0] + "_obj" + ".png")

        plt.figure(dpi=300)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.plot(self.err_list, label="error")
        plt.legend()
        plt.savefig(self.output_fig.split(".png")[0] + "_error" + ".png")
