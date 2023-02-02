import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

import qutip.logging_utils as logging
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen



def optcontrol(example_name, H_d, H_c, X_0, X_targ, n_ts, evo_time, initial_type, initial_control,
               output_num=None, output_fig=None, output_control=None, sum_cons_1=False, example='CNOT',
               fid_err_targ=1e-10, max_iter=500, max_wall_time=600, min_grad=1e-20, constant=0.5):
    """

    :param example_name: name of the example
    :param H_d: drift matrix
    :param H_c: list of control matrix
    :param n_ts: time steps
    :param evo_time: evolution time
    :param initial_type: type of initialization
    :param initial_control: the initial controls if use warm start
    :param output_num: output file of the numerical results
    :param output_fig: output file of the control figure
    :param output_control: output control file
    :param sum_one_constraint: if there are constraints of summation as one.
    :param fid_err_targ: Fidelity error target
    :param max_iter: maximum number of iterations
    :param: max_wall_time: maximum time allowed in seconds
    :param: min_grad: minimum gradient
    :param: offset: value of the constant initialization
    :return:
    """
    logger = logging.get_logger()
    # Set this to None or logging.WARN for 'quiet' execution
    log_level = logging.INFO

    # if sum_cons_1:
    #     # if there are constraints about the summation, we modify the control Hamiltonians
    #     # The control Hamiltonians
    #     H_c_origin = H_c
    #     H_c = [H_c_origin[i] - H_c_origin[-1] for i in range(len(H_c_origin) - 1)]
    #     # Drift Hamiltonian
    #     H_d = H_d + H_c_origin[-1]

    # Number of controllers
    n_ctrls = len(H_c)

    # Set initial state
    # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
    p_type = initial_type
    offset = constant
    # lower bound and upper bound of initial value
    init_lb = 0
    init_ub = 1
    obj_type = "UNIT"
    # Set output files
    # f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
    # Build the optimizer
    optim = cpo.create_pulse_optimizer(H_d, H_c, X_0, X_targ, n_ts, evo_time,
                                       amp_lbound=0, amp_ubound=1,
                                       fid_err_targ=fid_err_targ, min_grad=min_grad,
                                       max_iter=max_iter, max_wall_time=max_wall_time, dyn_type='UNIT',
                                       fid_type=obj_type, phase_option="PSU",
                                       init_pulse_params={"offset": offset},
                                       log_level=log_level, gen_stats=True)

    # Initialization
    dyn = optim.dynamics

    # Generate different initial pulses for each of the controls
    init_amps = np.zeros([n_ts, n_ctrls])
    for j in range(n_ctrls):
        if p_type == "RND":
            p_gen = pulsegen.create_pulse_gen(p_type, dyn)
            p_gen.lbound = init_lb
            p_gen.ubound = init_ub
            p_gen.offset = offset
            init_amps[:, j] = p_gen.gen_pulse()
        elif p_type == "CONSTANT":
            if isinstance(offset, float) or isinstance(offset, int):
                init_amps[:, j] = offset
            else:
                init_amps[:, j] = offset[j]
        else:
            file = open(initial_control)
            if sum_cons_1:
                warm_start_control = np.loadtxt(file, delimiter=",")[:, 0]
            else:
                warm_start_control = np.loadtxt(file, delimiter=",")
            evo_time_start = warm_start_control.shape[0]
            step = n_ts / evo_time_start
            for j in range(n_ctrls):
                for time_step in range(n_ts):
                    init_amps[time_step, j] = warm_start_control[int(np.floor(time_step / step)), j]
        dyn.initialize_controls(init_amps)

    # Run the optimization
    result = optim.run_optimization()

    # f = open("test.log","w+")
    # print(optim.dynamics.fwd_evo, file=f)
    # f.close()
    # Report the results
    if output_num:
        report = open(output_num, "w+")
        print("Final evolution\n{}\n".format(result.evo_full_final), file=report)
        print("********* Summary *****************", file=report)
        if example == "Leak":
            print("Final fidelity error {}".format(result.fid_err), file=report)
        else:
            print("Final fidelity error {}".format(result.fid_err), file=report)
        print("Final gradient normal {}".format(result.grad_norm_final), file=report)
        print("Terminated due to {}".format(result.termination_reason), file=report)
        print("Number of iterations {}".format(result.num_iter), file=report)
        print("Completed in {} HH:MM:SS.US".format(
            datetime.timedelta(seconds=result.wall_time)), file=report)
        result.stats.report()
        report.close()
    
    # f = open(output_num, "r")
    # dataset = f.readlines()
    # print(dataset)
    # Plot the results
    fig1 = plt.figure(dpi=300)
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_title("Initial control amps")
    # ax1.set_xlabel("Time")
    ax1.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax1.step(result.time,
                 np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
                 where='post')
    if sum_cons_1:
        ax1.step(result.time,
                 np.hstack((1 - result.initial_amps[:, 0], 1 - result.initial_amps[-1, 0])),
                 where='post')
    # if sum_one_constraint:
    #     ax1.step(result.time,
    #              np.hstack((1 - sum(result.initial_amps[:, j] for j in range(n_ctrls)),
    #                         1 - sum(result.initial_amps[-1, j] for j in range(n_ctrls)))),
    #              where='post')

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_title("Optimised Control Sequences")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax2.step(result.time,
                 np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
                 where='post')
    if sum_cons_1:
        ax1.step(result.time,
                 np.hstack((1 - result.final_amps[:, 0], 1 - result.final_amps[-1, 0])),
                 where='post')
    # if sum_one_constraint:
    #     ax2.step(result.time,
    #              np.hstack((1 - sum(result.final_amps[:, j] for j in range(n_ctrls)),
    #                         1 - sum(result.final_amps[-1, j] for j in range(n_ctrls)))),
    #              where='post')
    plt.tight_layout()
    if output_fig:
        plt.savefig(output_fig)

    final_amps = np.zeros((n_ts, max(n_ctrls, 2)))
    # if sum_one_constraint:
    #     final_amps = np.zeros((n_ts, n_ctrls + 1))
    for j in range(n_ctrls):
        final_amps[:, j] = result.final_amps[:, j]
    if sum_cons_1:
        final_amps[:, 1] = 1 - result.final_amps[:, 0]
    # if sum_one_constraint:
    #     for t in range(n_ts):
    #         final_amps[t, n_ctrls] = 1 - sum(result.final_amps[t, j] for j in range(n_ctrls))

    if output_control:
        np.savetxt(output_control, final_amps, delimiter=",")
