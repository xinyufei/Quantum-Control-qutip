import sys
import time
import numpy as np
import random
import qutip.control.pulseoptim as cpo
from qutip.control.optimizer import *
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

sys.path.append("..")
from tools import *
import gurobipy as gb


class TrustRegion:
    def __init__(self):
        self.H_d = None
        self.H_c = None
        self.X_0 = None
        self.X_targ = None
        self.n_ts = None
        self.evo_time = None
        self.delta_t = None
        self.obj_type = None
        self.n_ctrl = None
        self.optim = None

        self.u = None
        self._into = None
        self._onto = None

        self.initial_file = None
        self.initial_amps = None
        self.alpha = None
        self.sigma = None
        self.eta = None
        self.delta_threshold = None
        self.max_iter = None
        self.out_log_file = None
        self.out_control_file = None
        self.out_fig_file = None

    def build_optimizer(self, H_d, H_c, X_0, X_targ, n_ts, evo_time, alpha=1e-3, obj_type='fid',
                        initial_file=None, sigma=0.25, eta=1e-3, delta_threshold=30, max_iter=100,
                        out_log_file=None, out_control_file=None, out_fig_file=None):
        self.H_d = H_d
        self.H_c = H_c
        self.X_0 = X_0
        self.X_targ = X_targ
        self.n_ts = n_ts
        self.evo_time = evo_time
        self.delta_t = self.evo_time / self.n_ts
        self.obj_type = obj_type

        if self.obj_type == 'fid':
            self.n_ctrl = len(self.H_c)
            self.optim = cpo.create_pulse_optimizer(Qobj(self.H_d), [Qobj(hc) for hc in self.H_c],
                                                    Qobj(self.X_0), Qobj(self.X_targ), self.n_ts, self.evo_time,
                                                    amp_lbound=0, amp_ubound=1, dyn_type='UNIT', phase_option="PSU",
                                                    init_pulse_params={"offset": 0}, gen_stats=True)
        if self.obj_type == 'energy':
            self.n_ctrl = len(self.H_c) - 1
            self._into = None
            self._onto = None

        self.initial_file = initial_file
        self.alpha = alpha
        # if self.obj_type == 'energy':
        #     self.alpha = 2 * alpha
        self.sigma = sigma
        self.eta = eta
        self.delta_threshold = delta_threshold
        self.max_iter = max_iter
        self.out_log_file = out_log_file
        self.out_control_file = out_control_file
        self.out_fig_file = out_fig_file

    def _init_amps(self):
        # self.initial_amps = np.zeros((self.n_ts, self.n_ctrl))
        self.initial_amps = np.loadtxt(self.initial_file, delimiter=",")
        if self.obj_type == 'energy':
            self.initial_amps = np.expand_dims(self.initial_amps[:, 0], 1)
        if self.obj_type == 'fid':
            dyn = self.optim.dynamics
            dyn.initialize_controls(self.initial_amps)

    def time_evolution_energy(self, control_amps):
        self._into = [self.X_0]
        for k in range(self.n_ts):
            fwd = expm(-1j * (control_amps[k] * self.H_c[0] + (1 - control_amps[k]) * self.H_c[1]) * self.delta_t).dot(
                self._into[k])
            self._into.append(fwd)

    def back_propagation_energy(self, control_amps):
        self._onto = [self._into[-1].conj().T.dot(self.H_c[1].conj().T)]
        for k in range(self.n_ts):
            bwd = self._onto[k].dot(expm(-1j * (control_amps[self.n_ts - k - 1] * self.H_c[0] + (
                    1 - control_amps[self.n_ts - k - 1]) * self.H_c[1]) * self.delta_t))
            self._onto.append(bwd)

    def _compute_obj(self, control_amps):
        if self.obj_type == 'fid':
            obj = self.optim.fid_err_func_compute(control_amps)
        if self.obj_type == 'energy':
            if not (self.u == control_amps).all():
                self.time_evolution_energy(control_amps)
                self.back_propagation_energy(control_amps)
                self.u = control_amps
            obj = np.real(self._into[-1].conj().T.dot(self.H_c[1].dot(self._into[-1])))
        return obj

    def _compute_gradient(self, control_amps):
        if self.obj_type == 'fid':
            grad = self.optim.fid_err_grad_compute(control_amps)
        if self.obj_type == 'energy':
            if not (self.u == control_amps).all():
                self.time_evolution_energy(control_amps)
                self.back_propagation_energy(control_amps)
                self.u = control_amps
            grad = []
            for k in range(self.n_ts):
                grad += [-np.imag(self._onto[self.n_ts - k - 1].dot((self.H_c[1] - self.H_c[0]).dot(self._into[k + 1]))
                                  * self.delta_t)]
            grad = np.expand_dims(np.array(grad), 1)
        return grad

    def _compute_tv_norm(self, control_amps):
        norm = sum(sum(np.abs(control_amps[t, j] - control_amps[t + 1, j]) for j in range(self.n_ctrl))
                   for t in range(self.n_ts - 1))
        if self.obj_type == 'energy':
            norm = 2 * norm
        return norm

    def trust_region_method_tv(self, type='binary', sos1=1):
        delta_0 = self.n_ts * self.n_ctrl

        out_log = open(self.out_log_file, "w+")

        terminate = False
        total_ite = 0

        pred = np.infty

        self._init_amps()
        u_tilde = self.initial_amps.copy()

        start = time.time()
        for n in range(self.max_iter):
            k = 0
            delta_n = delta_0
            delta_list = []

            # get the gradient
            grad = self._compute_gradient(u_tilde.reshape(-1))
            # pre-compute objective value
            obj_u_tilde = self._compute_obj(u_tilde.reshape(-1))
            # pre-compute tv norm of u_tilde
            tv_u_tilde = self._compute_tv_norm(u_tilde)

            while 1:
                # solve the trust-region problem
                tr = gb.Model()
                # add variants
                if type == 'binary':
                    u_var = tr.addVars(self.n_ts, self.n_ctrl, vtype=gb.GRB.BINARY)
                if type == 'continuous':
                    u_var = tr.addVars(self.n_ts, self.n_ctrl, vtype=gb.GRB.CONTINUOUS)
                v_var = tr.addVars(self.n_ts - 1, self.n_ctrl, lb=0)
                w_var = tr.addVars(self.n_ts, self.n_ctrl, lb=0)
                # objective function
                tr.setObjective(gb.quicksum(gb.quicksum(grad[t, j] * (u_var[t, j] - u_tilde[t, j])
                                                        for j in range(self.n_ctrl)) for t in range(self.n_ts))
                                + self.alpha * gb.quicksum(gb.quicksum(v_var[t, j] for j in range(self.n_ctrl))
                                                           for t in range(self.n_ts - 1)) - self.alpha * tv_u_tilde)
                tr.addConstrs(u_var[t, j] - u_tilde[t, j] + w_var[t, j] >= 0 for t in range(self.n_ts)
                              for j in range(self.n_ctrl))
                tr.addConstrs(u_var[t, j] - u_tilde[t, j] - w_var[t, j] <= 0 for t in range(self.n_ts)
                              for j in range(self.n_ctrl))
                tr.addConstr(gb.quicksum(gb.quicksum(w_var[t, j] for j in range(self.n_ctrl))
                                         for t in range(self.n_ts)) <= delta_n)
                tr.addConstrs(u_var[t, j] - u_var[t + 1, j] + v_var[t, j] >= 0 for t in range(self.n_ts - 1)
                              for j in range(self.n_ctrl))
                tr.addConstrs(u_var[t, j] - u_var[t + 1, j] - v_var[t, j] <= 0 for t in range(self.n_ts - 1)
                              for j in range(self.n_ctrl))
                if self.n_ctrl > 1 and sos1 == 1:
                    tr.addConstrs(gb.quicksum(u_var[t, j] for j in range(self.n_ctrl)) == 1 for t in range(self.n_ts))
                # solve the optimization model
                tr.optimize()
                # obtain the optimal solution
                u_val = np.zeros((self.n_ts, self.n_ctrl))
                for t in range(self.n_ts):
                    for j in range(self.n_ctrl):
                        u_val[t, j] = u_var[t, j].x
                # u_val = tr.getAttr('X', u_var)
                # compute the TV norm
                tv_u_val = self._compute_tv_norm(u_val)
                # compute the predictive decrease
                pred = sum(sum(grad[t, j] * (u_tilde[t, j] - u_val[t, j]) for j in range(self.n_ctrl))
                           for t in range(self.n_ts)) + self.alpha * tv_u_tilde - self.alpha * tv_u_val
                # compute the actual decrease
                ared = obj_u_tilde + self.alpha * tv_u_tilde - self._compute_obj(u_val.reshape(-1)) \
                       - self.alpha * tv_u_val

                delta_list.append(delta_n)

                if pred <= 0:
                    # obtain the local minimum
                    terminate = True
                    break
                elif ared < self.sigma * pred:
                    # reduce the trust-region radius
                    k = k + 1
                    if delta_n <= self.delta_threshold:
                        if delta_n <= 1:
                            delta_n /= 2
                        else:
                            delta_n = delta_n - 1
                    else:
                        delta_n = max(delta_n / 2, self.delta_threshold)

                # if there is sufficient decrease
                if ared >= self.eta * pred:
                    u_tilde = u_val
                    tv_u_tilde = tv_u_val
                    k = k + 1
                    break

            total_ite += k

            out_log = open(self.out_log_file, "a+")
            print(delta_list, file=out_log)
            print("predictive decrease", pred, "actual decrease", ared, file=out_log)
            obj = self._compute_obj(u_tilde.reshape(-1))
            print("objective value without tv norm", obj, file=out_log)
            print("objective value with tv norm", obj + self.alpha * tv_u_tilde, file=out_log)
            out_log.close()

            if terminate:
                break

        end = time.time()

        obj = self._compute_obj(u_tilde.reshape(-1))

        out_log = open(self.out_log_file, "a+")
        print("objective value without tv norm", obj, file=out_log)
        print("alpha", self.alpha, file=out_log)
        print("objective value with tv norm", obj + self.alpha * tv_u_tilde, file=out_log)
        print("norm", tv_u_tilde, file=out_log)
        print("computational time", end - start, file=out_log)
        print("total iterations", total_ite, file=out_log)
        out_log.close()

        if self.obj_type == 'energy':
            final_u = np.zeros((self.n_ts, 2))
            final_u[:, 0] = u_tilde[:, 0]
            final_u[:, 1] = 1 - u_tilde[:, 0]
            np.savetxt(self.out_control_file, final_u, delimiter=',')
        else:
            np.savetxt(self.out_control_file, u_tilde, delimiter=',')

        return u_tilde, obj, obj + self.alpha * tv_u_tilde

    def trust_region_method_hard(self, cons_parameter=None, sos1=1):
        if cons_parameter is None:
            cons_parameter = dict(hard_type='minup', time=10)

        delta_0 = self.n_ts * self.n_ctrl

        out_log = open(self.out_log_file, "w+")

        terminate = False
        total_ite = 0

        pred = np.infty

        self._init_amps()
        u_tilde = self.initial_amps.copy()

        start = time.time()
        for n in range(self.max_iter):
            k = 0
            delta_n = delta_0
            delta_list = []

            # get the gradient
            grad = self._compute_gradient(u_tilde.reshape(-1))
            # pre-compute objective value
            obj_u_tilde = self._compute_obj(u_tilde.reshape(-1))

            while 1:
                # solve the trust-region problem
                tr = gb.Model()
                # add variants
                u_var = tr.addVars(self.n_ts, self.n_ctrl, vtype=gb.GRB.BINARY)
                # u_var = tr.addVars(n_ts, n_ctrl, vtype=gb.GRB.CONTINUOUS)
                v_var = tr.addVars(self.n_ts - 1, self.n_ctrl, lb=0)
                w_var = tr.addVars(self.n_ts, self.n_ctrl, lb=0)
                # objective function
                # pre-compute tv norm of u_tilde
                tr.setObjective(gb.quicksum(gb.quicksum(grad[t, j] * (u_var[t, j] - u_tilde[t, j])
                                                        for j in range(self.n_ctrl)) for t in range(self.n_ts)))
                tr.addConstrs(u_var[t, j] - u_tilde[t, j] + w_var[t, j] >= 0
                              for t in range(self.n_ts) for j in range(self.n_ctrl))
                tr.addConstrs(u_var[t, j] - u_tilde[t, j] - w_var[t, j] <= 0
                              for t in range(self.n_ts) for j in range(self.n_ctrl))
                tr.addConstr(gb.quicksum(
                    gb.quicksum(w_var[t, j] for j in range(self.n_ctrl)) for t in range(self.n_ts)) <= delta_n)
                tr.addConstrs(u_var[t, j] - u_var[t + 1, j] + v_var[t, j] >= 0
                              for t in range(self.n_ts - 1) for j in range(self.n_ctrl))
                tr.addConstrs(u_var[t, j] - u_var[t + 1, j] - v_var[t, j] <= 0
                              for t in range(self.n_ts - 1) for j in range(self.n_ctrl))
                if cons_parameter['hard_type'] == 'minup':
                    min_up_time = cons_parameter['time']
                    tr.addConstrs(gb.quicksum(v_var[t + tt, j] for tt in range(min_up_time)) <= 1
                                  for t in range(self.n_ts - min_up_time) for j in range(self.n_ctrl))
                    tr.addConstrs(gb.quicksum(v_var[t, j] for t in range(min_up_time - 1)) == 0
                                  for j in range(self.n_ctrl))
                if cons_parameter['hard_type'] == 'maxswitch':
                    max_switches = cons_parameter['switch']
                    tr.addConstrs(gb.quicksum(v_var[t, j] for t in range(self.n_ts - 1)) <= max_switches
                                  for j in range(self.n_ctrl))
                if self.n_ctrl > 1 and sos1 == 1:
                    tr.addConstrs(gb.quicksum(u_var[t, j] for j in range(self.n_ctrl)) == 1 for t in range(self.n_ts))
                # solve the optimization model
                tr.optimize()
                # obtain the optimal solution
                u_val = np.zeros((self.n_ts, self.n_ctrl))
                for t in range(self.n_ts):
                    for j in range(self.n_ctrl):
                        u_val[t, j] = u_var[t, j].x

                # compute the predictive decrease
                pred = sum(sum(grad[t, j] * (u_tilde[t, j] - u_val[t, j]) for j in range(self.n_ctrl))
                           for t in range(self.n_ts))
                # compute the actual decrease
                ared = obj_u_tilde - self._compute_obj(u_val.reshape(-1))

                delta_list.append(delta_n)
                delta_n_pre = delta_n

                if pred <= 0:
                    # obtain the local minimum
                    terminate = True
                    break
                elif ared < self.sigma * pred:
                    # reduce the trust-region radius
                    k = k + 1
                    if delta_n <= self.delta_threshold:
                        if delta_n <= 1:
                            delta_n /= 2
                        else:
                            delta_n = delta_n - 1
                    else:
                        delta_n = max(delta_n / 2, self.delta_threshold)

                # if there is sufficient decrease
                if ared >= self.eta * pred:
                    u_tilde = u_val
                    k = k + 1
                    break

            total_ite += k

            out_log = open(self.out_log_file, "a+")
            print(delta_list, file=out_log)
            print("predictive decrease", pred, "actual decrease", ared, file=out_log)
            obj = self._compute_obj(u_tilde.reshape(-1))
            print("objective value with out tv norm", obj, file=out_log)
            out_log.close()

            if terminate:
                break

        end = time.time()

        obj = self._compute_obj(u_tilde.reshape(-1))

        out_log = open(self.out_log_file, "a+")
        print("objective value without tv norm", obj, file=out_log)
        tv_u_tilde = self._compute_tv_norm(u_tilde)
        print("norm", tv_u_tilde, file=out_log)
        print("computational time", end - start, file=out_log)
        print("total iterations", total_ite, file=out_log)
        out_log.close()

        if self.obj_type == 'energy':
            final_u = np.zeros((self.n_ts, 2))
            final_u[:, 0] = u_tilde[:, 0]
            final_u[:, 1] = 1 - u_tilde[:, 0]
            np.savetxt(self.out_control_file, final_u, delimiter=',')
        else:
            np.savetxt(self.out_control_file, u_tilde, delimiter=',')

        return u_tilde, obj
