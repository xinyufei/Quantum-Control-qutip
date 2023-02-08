import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, expm_frechet

import qutip.logging_utils as logging
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

import scipy

from tools import *


class optcontrol_energy():
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
        self.u = None
        self.result = None

        self._into = None
        self._onto = None

    def build_optimizer(self, B, C, n, y0, n_ts, evo_time, initial_type, initial_control=None,
                        output_num=None, output_fig=None, output_control=None, max_iter=100, max_wall_time=600,
                        min_grad=0.05, constant=0.5):
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

        self.matrix_exp = []
        s_b, v_b = np.linalg.eigh(B)
        self.matrix_exp.append(np.dot(v_b.dot(np.diag(np.exp(-1j * s_b * self.delta_t))), v_b.conj().T))
        s_c, v_c = np.linalg.eigh(C)
        self.matrix_exp.append(np.dot(v_c.dot(np.diag(np.exp(-1j * s_c * self.delta_t))), v_c.conj().T))

    def time_evolution(self, control_amps):
        self._into = [self.y0]
        for k in range(self.n_ts):
            # fwd = expm(-1j * (control_amps[k] * self.B + (1 - control_amps[k]) * self.C) * self.delta_t).dot(
            #     self._into[k])
            fwd = (control_amps[k] * self.matrix_exp[0] + (1 - control_amps[k]) * self.matrix_exp[1]).dot(
                self._into[k])
            self._into.append(fwd)

    def back_propagation(self, control_amps):
        self._onto = [self._into[-1].conj().T.dot(self.C.conj().T)]
        for k in range(self.n_ts):
            # bwd = self._onto[k].dot(expm(-1j * (control_amps[self.n_ts - k - 1] * self.B + (
            #         1 - control_amps[self.n_ts - k - 1]) * self.C) * self.delta_t))
            bwd = self._onto[k].dot(control_amps[self.n_ts - k - 1] * self.matrix_exp[0] + (
                    1 - control_amps[self.n_ts - k - 1]) * self.matrix_exp[1])
            self._onto.append(bwd)

    def _compute_energy(self, *args):
        control_amps = args[0].copy()
        if not (self.u == control_amps).all():
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
            # grad += [calc_Phi(self.all_y[i], self.all_k[i], self.n, self.diag)]
            # grad += [-np.imag(self._onto[self.n_ts - k - 1].dot((self.C - self.B).dot(self._into[k + 1]))
            #                   * self.delta_t)]
            # grad += [-np.imag(self._onto[self.n_ts - k - 1].dot((self.C - self.B).dot(self._into[k + 1]))
            #                   * self.delta_t) * 2]
            grad += [np.imag(self._onto[self.n_ts - k - 1].dot(
                (self.matrix_exp[0] - self.matrix_exp[1]).dot(self._into[k]))) * 2]
        # print(grad)
        return grad

    def _set_initial_amps(self):
        self.initial_amps = np.zeros(self.n_ts)
        if self.initial_type == "RND":
            np.random.seed(5)
            self.initial_amps = np.random.random(self.n_ts)
            print(self.initial_amps)
        if self.initial_type == "CONSTANT":
            self.initial_amps = np.ones(self.n_ts) * self.constant
        if self.initial_type == "WARM":
            warm_start_control = np.loadtxt(self.initial_control, delimiter=",")[:, 0]
            evo_time_start = warm_start_control.shape[0]
            step = self.n_ts / evo_time_start
            for time_step in range(self.n_ts):
                self.initial_amps[time_step] = warm_start_control[int(np.floor(time_step / step))]

    def optimize(self):
        self._set_initial_amps()
        start = time.time()
        results = scipy.optimize.fmin_l_bfgs_b(self._compute_energy, self.initial_amps.copy(),
                                               bounds=[(0, 1)] * self.n_ts,
                                               pgtol=self.min_grad, fprime=self._fprime, maxiter=self.max_iter)
        # results = scipy.optimize.fmin_l_bfgs_b(self._compute_energy, self.initial_amps.copy(),
        #                                        bounds=[(0, 1)] * self.n_ts, approx_grad=1,
        #                                        pgtol=self.min_grad, maxiter=self.max_iter, iprint=101)
        # results = scipy.optimize.minimize(self._compute_energy, self.initial_amps.copy(),
        #                                   # jac=self._fprime,
        #                                   # method='SLSQP',
        #                                   bounds=[(0, 1)] * self.n_ts, tol=self.min_grad,
        #                                   options={'maxiter': self.max_iter})
        # [ulist, Philist, Energy, state, nit] = gradient_descent_opt(self.n, self.n_ts, self.evo_time, self.max_iter,
        #                                                             self.min_grad)
        # Energy = compute_energy_u(self.tlist, self.evo_time, self.u)
        end = time.time()
        self.result = results
        self.u = results[0]
        # self.u = np.expand_dims(results.x, 1)
        # self.u = np.expand_dims(np.array(self.u), 1)

        tr = None
        # if nit < self.max_iter - 1:
        if results[2]['warnflag'] == 0:
            tr = "Function converged"
        if results[2]['warnflag'] == 1:
            tr = "Exceed maximum iterations"
        if results[2]['warnflag'] == 2:
            tr = results[2]['task']

        report = open(self.output_num, "w+")
        print("********* Summary *****************", file=report)
        print("Final energy {}".format(results[1]), file=report)
        # print("Final gradient {}".format(Philist), file=report)
        print("Final gradient norm {}".format(np.linalg.norm(results[2]['grad'], 2)), file=report)
        print("Number of iterations {}".format(results[2]['nit']), file=report)
        print("Terminated due to {}".format(tr), file=report)
        print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=end - start)), file=report)
        print("Computational time {}".format(end - start), file=report)

        final_u = np.zeros((self.n_ts, 2))
        for k in range(self.n_ts):
            final_u[k, 0] = self.u[k]
            final_u[k, 1] = 1 - self.u[k]

        if self.output_control:
            np.savetxt(self.output_control, final_u, delimiter=",")

        # output the figures
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.set_title("Initial control amps")
        # ax1.set_xlabel("Time")
        ax1.set_ylabel("Control amplitude")
        ax1.step(self.tlist, np.hstack((self.initial_amps[0:self.n_ts], self.initial_amps[-2])), where='post')
        ax1.step(self.tlist, np.hstack((1 - self.initial_amps[0:self.n_ts], 1 - self.initial_amps[-2])),
                 where='post')

        ax2 = fig1.add_subplot(2, 1, 2)
        ax2.set_title("Optimised Control Sequences")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Control amplitude")
        for j in range(2):
            ax2.step(self.tlist, np.hstack((final_u[:, j], final_u[-1, j])), where='post')
        plt.tight_layout()
        if self.output_fig:
            plt.savefig(self.output_fig)
