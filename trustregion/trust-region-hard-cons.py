import time
import numpy as np
import random
import gurobipy as gb
import qutip.control.pulseoptim as cpo
from qutip.control.optimizer import *
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
from auxiliary_function import *


def trust_region_method_hard(H_d_qobj, H_c_qobj, X_0_qobj, X_targ_qobj, n_ts, evo_time, sigma, alpha, eta, max_iter):
    optim = cpo.create_pulse_optimizer(H_d_qobj, H_c_qobj, X_0_qobj, X_targ_qobj,
                                       n_ts, evo_time, amp_lbound=0, amp_ubound=1,
                                       dyn_type='UNIT', phase_option="PSU",
                                       init_pulse_params={"offset": 0}, gen_stats=True)
    # initialize trust-region radius and starting point
    n_ctrl = len(H_c_qobj)
    delta_0 = n_ts * n_ctrl
    u_0 = np.zeros((n_ts, n_ctrl))
    for t in range(n_ts):
        # u_0[t, 0] = 1
        # if random.random() < 0.5:
        #     u_0[t, 0] = 1
        # else:
        #     u_0[t, 1] = 1
        ctrl = random.randint(0, n_ctrl - 1)
        u_0[t, ctrl] = 1
    initial_file = "initial/Origin/" + "SPINWARMNEW5" \
                   + "_evotime20_n_ts200_ptypeWARM_offset0_objUNIT_penalty0_sum_penalty0.5_binary_1.csv"

    # initial_file = "initial/Origin/" + "SPINWARMNEW2" \
    #                + "_evotime8_n_ts80_ptypeWARM_offset0_objUNIT_penalty0_sum_penalty0.001_binary_1.csv"
    # initial_file = "initial/ADMM/" + "CNOTSUM1INEXACT" \
    #                + "_evotime10_n_ts200_ptypeWARM_offset0_objUNIT_penalty0.001_ADMM_0.25_iter200_binary_0.5.csv"
    u_0 = np.loadtxt(initial_file)

    # for t in range(n_ts - 10):
    #     for j in range(n_ctrl):
    #         if sum(np.abs(u_0[t + tt + 1, j] - u_0[t + tt, j]) for tt in range(10)) > 1:
    #             print("wrong!")
    # print(sum(u_0[t, 0] for t in range(n_ts)))
    # print(sum(u_0[t, 1] for t in range(n_ts)))
    u_tilde = u_0.copy()

    dyn = optim.dynamics
    dyn.initialize_controls(u_0)

    # out_log = open("tr-log/n_ts" + str(n_ts) + "_n_ctrl" + str(n_ctrl) + "continuous.log", "w+")
    # out_log.close()

    terminate = False
    # delta_threshold = n_ts * n_ctrl
    delta_threshold = 30
    total_ite = 0
    min_up_time = 10

    out_log_file = "tr-log-hard/" + initial_file.split("/")[2].split(".csv")[0] + "_sigma" + str(sigma) + "_eta" \
                   + str(eta) + "_threshold" + str(delta_threshold) + ".log"

    delta_n_pre = delta_0
    start = time.time()
    for n in range(max_iter):
        k = 0
        delta_n = max(min(delta_0, 2*delta_n_pre), delta_threshold)
        # delta_n_pre = delta_n
        # u_tilde = u_0.copy()
        # delta_list = [delta_0]
        delta_list = []

        while 1:
            # solve the trust-region problem
            tr = gb.Model()
            # get the gradient
            grad = optim.fid_err_grad_compute(u_tilde.reshape(-1))
            # add variants
            u_var = tr.addVars(n_ts, n_ctrl, vtype=gb.GRB.BINARY)
            # u_var = tr.addVars(n_ts, n_ctrl, vtype=gb.GRB.CONTINUOUS)
            # v_var = tr.addVar(n_ts - 1, n_ctrl, vtype=gb.GRB.BINARY)
            v_var = tr.addVars(n_ts - 1, n_ctrl, lb=0)
            w_var = tr.addVars(n_ts, n_ctrl, lb=0)
            # objective function
            # pre-compute tv norm of u_tilde
            tv_u_tilde = sum(
                sum(np.abs(u_tilde[t, j] - u_tilde[t + 1, j]) for j in range(n_ctrl)) for t in range(n_ts - 1))
            tr.setObjective(gb.quicksum(gb.quicksum(
                grad[t, j] * (u_var[t, j] - u_tilde[t, j]) for j in range(n_ctrl))
                                        for t in range(n_ts)))
            # add constraints
            # tr.addConstr(gb.quicksum(gb.quicksum(u_var[t, j] - u_tilde[t, j] for t in range(n_ts)) for j in range(n_ctrl))
            #              <= delta_n)
            # tr.addConstr(gb.quicksum(gb.quicksum(u_var[t, j] - u_tilde[t, j] for t in range(n_ts)) for j in range(n_ctrl))
            #              >= -delta_n)
            tr.addConstrs(u_var[t, j] - u_tilde[t, j] + w_var[t, j] >= 0 for t in range(n_ts) for j in range(n_ctrl))
            tr.addConstrs(u_var[t, j] - u_tilde[t, j] - w_var[t, j] <= 0 for t in range(n_ts) for j in range(n_ctrl))
            tr.addConstr(gb.quicksum(gb.quicksum(w_var[t, j] for j in range(n_ctrl)) for t in range(n_ts)) <= delta_n)
            tr.addConstrs(u_var[t, j] - u_var[t + 1, j] + v_var[t, j] >= 0 for t in range(n_ts - 1) for j in range(n_ctrl))
            tr.addConstrs(
                u_var[t, j] - u_var[t + 1, j] - v_var[t, j] <= 0 for t in range(n_ts - 1) for j in range(n_ctrl))
            tr.addConstrs(
                gb.quicksum(v_var[t + tt, j] for tt in range(min_up_time)) <= 1 for t in range(n_ts - min_up_time) for j
                in range(n_ctrl))
            tr.addConstrs(gb.quicksum(v_var[t, j] for t in range(9)) == 0 for j in range(n_ctrl))
            tr.addConstrs(gb.quicksum(u_var[t, j] for j in range(n_ctrl)) == 1 for t in range(n_ts))
            # solve the optimization model
            tr.optimize()
            # obtain the optimal solution
            u_val = np.zeros((n_ts, n_ctrl))
            for t in range(n_ts):
                for j in range(n_ctrl):
                    u_val[t, j] = u_var[t, j].x
            # u_val = tr.getAttr('X', u_var)
            # compute the TV norm
            tv_u_val = sum(sum(np.abs(u_val[t, j] - u_val[t + 1, j]) for j in range(n_ctrl)) for t in range(n_ts - 1))
            # compute the predictive decrease
            pred = sum(sum(grad[t, j] * (u_tilde[t, j] - u_val[t, j]) for j in range(n_ctrl))
                       for t in range(n_ts))
            # compute the actual decrease
            ared = optim.fid_err_func_compute(np.reshape(u_tilde, -1)) \
                   - optim.fid_err_func_compute(np.reshape(u_val, -1))

            delta_list.append(delta_n)
            delta_n_pre = delta_n

            if pred <= 0:
                # obtain the local minimum
                terminate = True
                break
            elif ared < sigma * pred:
                # reduce the trust-region radius
                k = k + 1
                # delta_n = delta_n - 1  # todo: test different ways of updating radius
                # delta_n = max(delta_n - 5, delta_n / 2)
                # delta_threshold = delta_n_pre
                if delta_n <= delta_threshold:
                    delta_n = delta_n - 1
                else:
                    delta_n = max(delta_n / 2, delta_threshold)
                # delta_n_pre = 400
                # if ared / pred < 1/4:
                #     delta_n = delta_n / 4
                # else:
                #     delta_n = min(2*delta_n, n_ts * n_ctrl)
            # else:
            # u_tilde = u_val
            # k = k + 1

            # if there is sufficient decrease
            if ared >= eta * pred:
                u_tilde = u_val
                k = k + 1
                break

            # if ared >= sigma * pred:
            #     break

        total_ite += k

        # out_log = open("tr-log-hard/n_ts" + str(n_ts) + "_n_ctrl" + str(n_ctrl) + "_alpha" + str(alpha) + "_sigma"
        #                 + str(sigma) + "_eta" + str(eta) + "initial_0.5.log", "a+")

        out_log = open(out_log_file, "a+")
        print(delta_list, file=out_log)
        print("predictive decrease", pred, "actual decrease", ared, file=out_log)
        obj = optim.fid_err_func_compute(np.reshape(u_val, -1))
        print("objective value without tv norm", obj, file=out_log)
        print("objective value with tv norm", obj + alpha * tv_u_val, file=out_log)
        out_log.close()

        if terminate:
            break

    end = time.time()

    obj = optim.fid_err_func_compute(np.reshape(u_val, -1))

    # out_log = open("tr-log-hard/n_ts" + str(n_ts) + "_n_ctrl" + str(n_ctrl) + "_alpha" + str(alpha) + "_sigma"
    #                     + str(sigma) + "_eta" + str(eta) + "initial_0.5.log", "a+")
    out_log = open(out_log_file, "a+")
    np.savetxt("tr-log-hard/" + initial_file.split("/")[2].split(".csv")[0] + "_sigma" + str(sigma) + "_eta"
               + str(eta) + "_threshold" + str(delta_threshold) + ".csv", u_val)
    # np.savetxt("tr-log-hard/n_ts" + str(n_ts) + "_n_ctrl" + str(n_ctrl) + "_alpha" + str(alpha) + "_sigma" + str(sigma)
    #             + "_eta" + str(eta) + "_0.5_2.tsv", u_val)
    # np.savetxt("tr-log-hard/n_ts" + str(n_ts) + "_n_ctrl" + str(n_ctrl) + "_alpha" + str(alpha) + "_sigma" + str(sigma)
    #             + "_eta" + str(eta) + "initial_0.5_2.tsv", u_0)
    print("objective value without tv norm", obj, file=out_log)
    print("objective value with tv norm", obj + alpha * tv_u_val, file=out_log)
    print("computational time", end - start, file=out_log)
    print("total iterations", total_ite, file=out_log)
    out_log.close()

    return u_val, obj, obj + tv_u_val


if __name__ == '__main__':
    # H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]  # The control Hamiltonians (Qobj classes)
    # H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())  # Drift Hamiltonian
    # X_0 = identity(4)  # start point for the gate evolution
    # X_targ = cnot()  # Target for the gate evolution

    qubit_num = 5
    Hops, H0, U0, U = generate_spin_func(qubit_num)
    H_c = [Qobj(hops) for hops in Hops]
    H_d = Qobj(H0)
    X_0 = Qobj(U0)
    X_targ = Qobj(U)

    # evo_time = 10  # Time allowed for the evolution
    # n_ts = 20 * evo_time  # Number of time steps

    evo_time = qubit_num * 4  # Time allowed for the evolution
    n_ts = 10 * evo_time  # Number of time steps

    sigma = 0.25
    alpha = 0.01
    eta = 1e-3
    max_iter = 100
    trust_region_method(H_d, H_c, X_0, X_targ, n_ts, evo_time, sigma, alpha, eta, max_iter)
