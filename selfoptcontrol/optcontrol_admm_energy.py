import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

import qutip.logging_utils as logging
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

import scipy

from tools import *


class optcontrol_admm_energy():
    def __init__(self):
        self.J = None
        self.diag = None
        self.B = None
        self.C = None
        self.n = None
        self.y0 = None
        self.n_ts = None
        self.evo_time = None
        self.delta_t = None
        self.tlist = None
        self.initial_type = None
        self.initial_control = None
        self.initial_amps = None
        self.constant = 0.5

        self.output_num = None
        self.output_fig = None
        self.output_control = None

        self.max_iter = 200
        self.max_wall_time = 600
        self.min_grad = 0.05

        self.all_y = None
        self.all_k = None
        self.energy = None
        self.Philist = None
        self.u = None
        self.result = None

        self.cur_obj = 0
        self.cur_origin_obj = 0
        self.v = None
        self._lambda = None
        self.rho = None
        self.alpha = None
        self.err_list = []
        self.obj_list = []
        self.max_iter_admm = None
        self.max_wall_time_admm = None

        self.time_optimize_start_step = 0
        self._into = None
        self._onto = None

    def build_optimizer(self, B, C, n, y0, n_ts, evo_time, initial_type, initial_control=None,
                        output_num=None, output_fig=None, output_control=None, max_iter=100, max_iter_admm=100,
                        max_wall_time=600, max_wall_time_admm=np.infty, min_grad=0.05, constant=0.5, rho=0.25,
                        alpha=0.01):
        # self.J = Jij.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.n = B.shape[0]
        # self.diag = get_diag().copy()
        self.n = n
        self.y0 = y0
        self.n_ts = n_ts
        self.evo_time = evo_time
        self.delta_t = self.evo_time / self.n_ts
        self.tlist = list(map(lambda x: self.evo_time * x, map(lambda x: x / self.n_ts, range(0, self.n_ts + 1))))
        self.initial_type = initial_type
        self.initial_control = initial_control
        self.constant = constant

        self.output_num = output_num
        self.output_fig = output_fig
        self.output_control = output_control

        self.max_iter = max_iter
        self.max_wall_time = max_wall_time
        self.min_grad = min_grad

        self.rho = rho
        self.alpha = alpha * 2
        self.u = np.zeros(self.n_ts)
        self.v = np.zeros(self.n_ts - 1)
        self._lambda = np.zeros(self.n_ts - 1)

        self.cur_obj = 0
        self.max_wall_time_admm = max_wall_time_admm
        self.max_iter_admm = max_iter_admm

    def time_evolution(self, control_amps):
        self._into = [self.y0]
        for k in range(self.n_ts):
            fwd = expm(-1j * (control_amps[k] * self.B + (1 - control_amps[k]) * self.C) * self.delta_t).dot(
                self._into[k])
            self._into.append(fwd)

    def back_propagation(self, control_amps):
        self._onto = [self._into[-1].conj().T.dot(self.C.conj().T)]
        for k in range(self.n_ts):
            bwd = self._onto[k].dot(expm(-1j * (control_amps[self.n_ts - k - 1] * self.B + (
                    1 - control_amps[self.n_ts - k - 1]) * self.C) * self.delta_t))
            self._onto.append(bwd)

    def _compute_energy(self, *args):
        control_amps = args[0].copy()
        # if not (self.u == control_amps).all():
        self.time_evolution(control_amps)
        self.back_propagation(control_amps)
        self.u = control_amps
        obj = np.real(self._into[-1].conj().T.dot(self.C.dot(self._into[-1])))
        # return np.real(avg_energy(self.all_y[-1], self.diag))
        return obj

    def _fprime(self, *args):
        control_amps = args[0].copy()
        if not (self.u == control_amps).all():
            # self.all_y = odeint(
            #     func_schro, self.y0, self.tlist, args=(self.n, self.n_ts, self.evo_time, control_amps, self.diag))
            # self.all_k = get_k(
            #     self.all_y[-1], self.tlist, self.n, self.n_ts, self.evo_time, control_amps, self.diag)
            self.time_evolution(control_amps)
            self.back_propagation(control_amps)
            self.u = control_amps
        grad = []
        for k in range(self.n_ts):
            if k == 0:
                norm_grad_t = -self.rho * (control_amps[1] - control_amps[0] - self.v[0] + self._lambda[0])
            if k == self.n_ts - 1:
                norm_grad_t = self.rho * (control_amps[k] - control_amps[k - 1] - self.v[k - 1] + self._lambda[k - 1])
            if 0 < k < self.n_ts - 1:
                norm_grad_t = self.rho * (control_amps[k] - control_amps[k - 1] - self.v[k - 1] + self._lambda[k - 1]
                                          - (control_amps[k + 1] - control_amps[k] - self.v[k] + self._lambda[k]))

            # grad += [-np.imag(self._onto[self.n_ts - k - 1].dot((self.C - self.B).dot(self._into[k + 1]))
            #                   * self.delta_t) + norm_grad_t]
            grad += [-np.imag(self._onto[self.n_ts - k - 1].dot((self.C - self.B).dot(self._into[k + 1]))
                              * self.delta_t) * 2 + norm_grad_t]

        return grad

    def _set_initial_amps(self):
        self.initial_amps = np.zeros(self.n_ts)
        if self.initial_type == "RND":
            self.initial_amps = np.random.random(self.n_ts)
        if self.initial_type == "CONSTANT":
            self.initial_amps = np.ones(self.n_ts) * self.constant
        if self.initial_type == "WARM":
            warm_start_control = np.loadtxt(self.initial_control, delimiter=",")[:, 0]
            evo_time_start = warm_start_control.shape[0]
            step = self.n_ts / evo_time_start
            for time_step in range(self.n_ts):
                self.initial_amps[time_step] = warm_start_control[int(np.floor(time_step / step))]
        if self.initial_type == "ADMM":
            self.initial_amps = self.u.copy()

    def compute_norm(self):
        norm = sum(np.power(self.u[time_step + 1] - self.u[time_step] - self.v[time_step]
                            + self._lambda[time_step], 2) for time_step in range(self.n_ts - 1))
        return norm

    def compute_tv_norm(self):
        return sum(abs(self.u[t + 1] - self.u[t]) for t in range(self.n_ts - 1))

    def _minimize_u(self):
        self._set_initial_amps()
        self.time_optimize_start_step = 0
        # [ulist, self.Philist, self.energy, state, nit] = gradient_descent_opt(
        #     self.n, self.n_ts, self.evo_time, self.max_iter, self.min_grad, ulist_in=self.initial_amps,
        #     type="admm", v=self.v, rho=self.rho, _lambda=self._lambda)
        results = scipy.optimize.fmin_l_bfgs_b(self._compute_energy, self.initial_amps.copy(),
                                               bounds=[(0, 1)] * self.n_ts,
                                               pgtol=self.min_grad, fprime=self._fprime, maxiter=self.max_iter)
        self.u = results[0]
        self.energy = results[1]
        self.cur_obj = self.energy + self.rho / 2 * self.compute_norm()
        self.cur_origin_obj = self.energy + self.alpha * self.compute_tv_norm()

    def _minimize_v(self):
        for t in range(self.n_ts - 1):
            temp = self.u[t + 1] - self.u[t] + self._lambda[t]
            if temp > self.alpha / self.rho:
                self.v[t] = -self.alpha / self.rho + temp
            if temp < -self.alpha / self.rho:
                self.v[t] = self.alpha / self.rho + temp
            if -self.alpha / self.rho <= temp <= self.alpha / self.rho:
                self.v[t] = 0

    def _update_dual(self):
        for t in range(self.n_ts - 1):
            self._lambda[t] += self.u[t + 1] - self.u[t] - self.v[t]

    def _admm_err(self):
        err = sum(np.power(self.u[t + 1] - self.u[t] - self.v[t], 2) for t in range(self.n_ts - 1))
        return err

    def optimize_admm(self):
        self._set_initial_amps()
        init_amps = self.initial_amps.copy()
        self.v = np.zeros(self.n_ts - 1)
        self._lambda = np.zeros(self.n_ts - 1)
        self.initial_type = "ADMM"
        admm_start = time.time()
        admm_num_iter = 0
        time_iter = [0]
        threshold = 1e-1
        while 1:
            admm_num_iter += 1
            if admm_num_iter > 1:
                self._set_initial_amps()
            self._minimize_u()
            self._minimize_v()
            self._update_dual()
            err = self._admm_err()
            self.err_list.append(err)
            self.obj_list.append(self.cur_obj)
            admm_opt_time = time.time()
            time_iteration = admm_opt_time - admm_start - time_iter[-1]
            time_iter.append(time_iteration)
            if admm_opt_time - admm_start >= self.max_wall_time_admm:
                tr = "Exceed the max wall time of ADMM"
                break
            if admm_num_iter >= self.max_iter_admm:
                tr = "Exceed the maximum number of iteration of ADMM"
                break

        report = open(self.output_num, "w+")
        print("********* Summary *****************", file=report)
        print("Final energy {}".format(self.energy), file=report)
        print("Final objective value {}".format(self.cur_origin_obj), file=report)
        print("Final penalized TV regularizer {}".format(self.alpha * self.compute_tv_norm()), file=report)
        print("Final norm value {}".format(2 * self.compute_tv_norm()), file=report)
        print("Final error {}".format(self.err_list[-1]), file=report)
        # print("Final gradient {}".format(Philist), file=report)
        # print("Final gradient norm {}".format(np.linalg.norm(np.array(self.Philist), 2)), file=report)
        print("Number of iterations {}".format(admm_num_iter), file=report)
        print("Terminated due to {}".format(tr), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=admm_opt_time - admm_start)), file=report)
        print("Computational time {}".format(admm_opt_time - admm_start), file=report)

        final_u = np.zeros((self.n_ts, 2))
        for k in range(self.n_ts):
            final_u[k, 0] = self.u[k]
            final_u[k, 1] = 1 - self.u[k]

        if self.output_control:
            np.savetxt(self.output_control, final_u, delimiter=",")
        #
        # output the figures
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.set_title("Initial control amps")
        # ax1.set_xlabel("Time")
        ax1.set_ylabel("Control amplitude")
        ax1.step(self.tlist, np.hstack((init_amps[0:self.n_ts], init_amps[-2])), where='post')
        ax1.step(self.tlist, np.hstack((1 - init_amps[0:self.n_ts], 1 - init_amps[-2])), where='post')

        ax2 = fig1.add_subplot(2, 1, 2)
        ax2.set_title("Optimised Control Sequences")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Control amplitude")
        for j in range(2):
            ax2.step(self.tlist, np.hstack((final_u[:, j], final_u[-1, j])), where='post')
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
