import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def draw_stats():
    x = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    sum_norm = [167.48180578702548, 167.47867920029202, 95.40499099227462, 34.144492624667244,
                30.14356912948749, 18.70981873414983, 0.3962903548156124, 5.55381795807922e-07,
                8.428231286969557e-10]
    # sum_norm = [4980.880077104522, 603.8538414769712, 332.6922609756568, 311.7450060080427,
    #             176.5722320180486, 5.863445830245192e-06, 6.282453886269812e-07,
    #             6.174458598795899e-10, 1.8027156950348233e-11]
                # 8.822329350354554e-08, 1.4531085324239638e-08]

    # exit()
    # plt.plot(np.array(x), np.array(sum_norm), '-o', label='squared_L2_norm')
    # plt.plot(np.array(x), 1e-9 / np.power(np.array(x), 2))
    # plt.show()
    # exit()

    # plt.figure(dpi=300)
    # plt.plot(np.array(x), np.array(sum_norm), '-o', label='squared_L2_norm')
    # plt.xlabel("Penalty parameter")
    # plt.ylabel("Squared Penalized Term")
    # plt.legend()
    # # plt.savefig("../example/figure/Continuous/MoleculeNew_H2_evotime4.0_n_ts80.png")
    # plt.savefig("../example/figure/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200.png")

    constant_idx = 5
    bound = [sum_norm[constant_idx] * x[constant_idx] / rho for rho in x[1:]]
    plt.figure(dpi=300)
    plt.plot(np.log10(np.array(x[1:])), np.log10(np.array(sum_norm[1:])), '-o',
             label=r'Common logarithm of squared $L_2$ norm')
    matplotlib.rcParams['text.usetex'] = True
    plt.plot(np.log10(np.array(x[1:])), np.log10(np.array(bound)), '--',
             # label=r'$\log_{10}$' + str(round(sum_norm[constant_idx], 2)) + r'$-\log_{10} \rho$')
             label=r'$-$' + str(round(-np.log10(sum_norm[constant_idx] * x[constant_idx]), 2)) + r'$-\log_{10} \rho$')
    plt.xlabel("Common logarithm of penalty parameter")
    plt.ylabel("Common logarithm of squared penalized Term")
    plt.legend(loc="lower left")
    plt.savefig("../figure_paper/MoleculeNew_H2_evotime4.0_n_ts80_log10_wb.png")
    # plt.savefig("../figure_paper/MoleculeVQE_LiH_evotime20.0_n_ts200_log10_wb.png")


def draw_control(evo_time, n_ts, control, output_fig):
    plt.figure(dpi=300)
    plt.xlabel("Time")
    plt.ylabel("Control amplitude")
    plt.ylim([0, 1])
    marker_list = ['-o', '--^', '-*', '--s', '-P']
    marker_size_list = [5, 5, 8, 5, 8]
    for j in range(control.shape[1]):
        plt.step(np.linspace(0, evo_time, n_ts + 1), np.hstack((control[:, j], control[-1, j])), marker_list[j],
                 where='post', linewidth=2, label='controller ' + str(j + 1), markevery=5,
                 markersize=marker_size_list[j])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_fig)


def draw_sos1(n_ts, control, output_fig):
    plt.figure(dpi=300)
    plt.xlabel("Time")
    plt.ylabel("Absolute violation")
    plt.plot(np.linspace(0, evo_time, n_ts), [abs(sum(control[k, :]) - 1) for k in range(n_ts)],
             label='absolute violation')
    plt.legend()
    plt.savefig(output_fig)


def draw_integral_error(example="H2", ub=False):
    if example == "H2":
        evo_time = 4.0
        ts_list = [40, 80, 160, 240, 320, 400]
        delta_t_list = [evo_time / n_ts for n_ts in ts_list]
        integral_err = []
        if ub:
            ub_list = []
        for n_ts in ts_list:
            c_control_name = "../example/control/Continuous/MoleculeNew_H2_evotime4.0_n_ts" + str(n_ts) + \
                             "_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv"
            c_control = np.loadtxt(c_control_name, delimiter=',')
            b_control_name = "../example/control/Rounding/" + c_control_name.split('/')[-1].split('.csv')[0] + \
                             "_1_SUR.csv"
            b_control = np.loadtxt(b_control_name, delimiter=',')

            n_ctrls = c_control.shape[1]
            delta_t = evo_time / n_ts
            integral = np.zeros((n_ts + 1, n_ctrls))
            for j in range(n_ctrls):
                for t in range(n_ts):
                    integral[t + 1, j] = integral[t, j] + (c_control[t, j] - b_control[t, j]) * delta_t
            integral_err.append(np.max(abs(integral)))
            if ub:
                print(np.sum(integral, 1))
                epsilon = np.max(abs(np.sum(integral, 1)))
                ub_list.append((n_ctrls - 1) * delta_t + (2 * n_ctrls - 1) / n_ctrls * epsilon)
        # draw the figure
        plt.figure(dpi=300)
        plt.xlabel("Time steps")
        # plt.xlabel("Unit interval length")
        # plt.ylabel("Maximum integral error")
        # plt.plot(ts_list, integral_err, label='Maximum integral error')
        plt.ylabel("Common logarithm of maximum integral error")
        plt.plot(ts_list, np.log10(integral_err), '-o', label='Common logarithm of maximum integral error')
        if ub:
            plt.plot(ts_list, np.log10(ub_list), "--", label="Upper bound of common logarithm of integral error")
        # plt.plot(delta_t_list, integral_err, label='Maximum integral error')
        plt.legend()
        if ub:
            plt.savefig("../figure_paper/MoleculeNew_H2_evotime4.0_sur_error_delta_t_withub_log10.png")
        else:
            plt.savefig("../figure_paper/MoleculeNew_H2_evotime4.0_sur_error_delta_t_log10.png")

    if example == "LiH":
        evo_time = 20.0
        ts_list = [100, 200, 400, 600, 800, 1000]
        delta_t_list = [evo_time / n_ts for n_ts in ts_list]
        integral_err = []
        if ub:
            ub_list = []
        for n_ts in ts_list:
            c_control_name = "../example/control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts" + str(n_ts) + \
                             "_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv"
            c_control = np.loadtxt(c_control_name, delimiter=',')
            b_control_name = "../example/control/Rounding/" + c_control_name.split('/')[-1].split('.csv')[0] + \
                             "_1_SUR.csv"
            b_control = np.loadtxt(b_control_name, delimiter=',')

            n_ctrls = c_control.shape[1]
            delta_t = evo_time / n_ts
            integral = np.zeros((n_ts + 1, n_ctrls))
            for j in range(n_ctrls):
                for t in range(n_ts):
                    integral[t + 1, j] = integral[t, j] + (c_control[t, j] - b_control[t, j]) * delta_t
            integral_err.append(np.max(abs(integral)))
            if ub:
                print(np.sum(integral, 1))
                epsilon = np.max(abs(np.sum(integral, 1)))
                ub_list.append((n_ctrls - 1) * delta_t + (2 * n_ctrls - 1) / n_ctrls * epsilon)
        # draw the figure
        plt.figure(dpi=300)
        plt.xlabel("Time steps")
        # plt.xlabel("Unit interval length")
        # plt.ylabel("Maximum integral error")
        # plt.plot(ts_list, integral_err, label='Maximum integral error')
        # # plt.plot(delta_t_list, integral_err, label='Maximum integral error')
        # if ub:
        #     plt.plot(ts_list, ub_list, "--", label="Upper bound of integral error")
        plt.ylabel("Common logarithm of maximum integral error")
        plt.plot(ts_list, np.log10(integral_err), '-o', label='Common logarithm of maximum integral error')
        if ub:
            plt.plot(ts_list, np.log10(ub_list), "--", label="Upper bound of common logarithm of integral error")
        # plt.plot(delta_t_list, integral_err, label='Maximum integral error')
        plt.legend()
        if ub:
            plt.savefig("../figure_paper/MoleculeVQE_LiH_evotime20.0_sur_error_delta_t_withub_log10.png")
        else:
            plt.savefig("../figure_paper/MoleculeVQE_LiH_evotime20.0_sur_error_delta_t_log10.png")


def draw_obj_energy_c():
    # plt.figure(figsize=(15, 3), dpi=300)
    plt.figure(dpi=300)
    step = 1
    instance = np.array([1, 2, 3]) * step
    instance_name = ["Energy2", "Energy4", "Energy6"]
    # grape_ar = [0.999, 3.999/4, 3.868/4.886]
    # tr_ar = [0.999, 3.9939/4, 3.838/4.886]
    # admm_ar = [0.999, 3.993/4, 3.838/4.886]
    grape_tv = [0.989, 3.953, 3.808]
    tr_tv = [0.993, 3.962, 3.820]
    admm_tv = [0.999, 3.993, 3.806]
    width = 0.1 * step
    plt.bar(instance-width, grape_tv, alpha=0.9, width=width, hatch='/', color='lightgray', edgecolor='black', label='GRAPE')
    plt.bar(instance, tr_tv, alpha=0.9, width=width, hatch='\\', color='lightgray', edgecolor='black', label='TR')
    plt.bar(instance+width, admm_tv, alpha=0.9, width=width, hatch='+', color='lightgray', edgecolor='black', label='ADMM')
    x_loc = plt.MultipleLocator(step)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Energy_continuous_TV.png")


def draw_obj_energy_r():
    # plt.figure(figsize=(15, 3), dpi=300)
    plt.figure(dpi=300)
    step = 1
    instance = np.array([1, 2, 3]) * step
    instance_name = ["Energy2", "Energy4", "Energy6"]
    grape_sur_tv = [0.459, 3.578, 3.466]
    tr_sur_tv = [0.455, 3.619, 3.418]
    admm_sur_tv = [0.519, 3.534, 3.305]
    grape_sur_tv_alb = [0.897, 3.578, 3.522]
    tr_sur_tv_alb = [0.915, 3.619, 3.450]
    admm_sur_tv_alb = [0.957, 3.534, 3.340]
    tr_st = [0, 3.937, 3.330]
    admm_st = [0, 3.270, 3.036]

    grape_mt = [0.841, 3.884, 1.548]
    tr_mt = [0.841, 3.884, 1.868]
    admm_mt = [0.841, 3.884, 1.910]
    grape_mt_alb = [0.997, 3.884, 2.709]
    tr_mt_alb = [0.997, 3.884, 2.709]
    admm_mt_alb = [0.959, 3.84, 2.614]
    tr_st_mt = [0, 3.934, 3.036]
    admm_st_mt = [0, 3.934, 3.036]

    grape_ms = [0.971, 3.920, 3.706]
    tr_ms = [0.960, 3.926, 3.651]
    admm_ms = [0.972, 3.928, 3.730]
    grape_ms_alb = [0.999, 3.962, 3.789]
    tr_ms_alb = [0.997, 3.975, 3.781]
    admm_ms_alb = [0.998, 3.971, 3.752]

    method_list = [grape_sur_tv, tr_sur_tv, admm_sur_tv, grape_sur_tv_alb, tr_sur_tv_alb, admm_sur_tv_alb,
                   tr_st, admm_st, grape_mt, tr_mt, admm_mt, grape_mt_alb, tr_mt_alb, admm_mt_alb, tr_st_mt,
                   admm_st_mt, grape_ms, tr_ms, admm_ms, grape_ms_alb, tr_ms_alb, admm_ms_alb]

    width = 0.04 * step
    for i in range(11):
        plt.bar(instance - (11 - i) * width, method_list[i], width=width, edgecolor='black')
    for i in range(11, 22):
        plt.bar(instance + (i - 11) * width, method_list[i], width=width)
    # plt.bar(instance - width, grape_tv, alpha=0.9, width=width, hatch='/', color='lightgray', edgecolor='black',
    #         label='GRAPE+SUR')
    # plt.bar(instance, tr_tv, alpha=0.9, width=width, hatch='\\', color='lightgray', edgecolor='black', label='TR+SUR')
    # plt.bar(instance + width, admm_tv, alpha=0.9, width=width, hatch='+', color='lightgray', edgecolor='black',
    #         label='ADMM+SUR')
    x_loc = plt.MultipleLocator(step)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Energy_binary.png")


def draw_sur_improve():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_st = np.array([1, 2, 3, 8, 9])
    instance_l2 = np.array([8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_sur_tv = [0.897, 0.785204688, 0.720764602, 0.734, 0.969, 0.973, 0.952, 0.977, 0.545]
    pgrape_sur_tv = [0.973, 0.553]
    tr_sur_tv = [0.915, 0.797010362, 0.705988294, 0.724, 0.964, 0.970, 0.952, 0.955, 0.539]
    admm_sur_tv = [0.957, 0.790725327, 0.683639638, 0.734, 0.979, 0.973, 0.955, 0.918, 0.715]
    tr_st_tv = [0, 0.830134194, 0.669190782, 0.987, 0.951]
    admm_st_tv = [0, 0.622728045, 0.669190782, 0.865, 0.001]

    width = 0.15
    # plt.bar(np.array([1 - width * 2, 2 - width * 2, 3 - width * 2, 4 - width, 5 - width, 6 - width, 7 - width,
    #                   8 - width / 2 * 5, 9 - width / 2 * 5]), grape_sur_tv, alpha=0.9, width=width,
    #         hatch='/', edgecolor='black', label='GRAPE+SUR+ALB')
    # plt.bar(np.array([1 - width, 2 - width, 3 - width, 4, 5, 6, 7,
    #                   8 - width / 2 * 3, 9 - width / 2 * 3]), tr_sur_tv, alpha=0.9, width=width,
    #         hatch='\\', edgecolor='black', label='TR+SUR+ALB')
    # plt.bar(np.array([1, 2, 3, 4 + width, 5 + width, 6 + width, 7 + width,
    #                   8 - width / 2, 9 - width / 2]), admm_sur_tv, alpha=0.9, width=width,
    #         hatch='+', edgecolor='black', label='ADMM+SUR+ALB')
    # plt.bar(np.array([8 + width / 2, 9 + width / 2]), pgrape_sur_tv, alpha=0.9, width=width,
    #         hatch='o', edgecolor='black', label='p-GRAPE+SUR+ALB')
    # plt.bar(np.array([1 + width, 2 + width, 3 + width, 8 + width / 2 * 3, 9 + width / 2 * 3]), tr_st_tv, alpha=0.9,
    #         width=width, hatch='.', edgecolor='black', label='TR+ST')
    # plt.bar(np.array([1 + width * 2, 2 + width * 2, 3 + width * 2, 8 + width / 2 * 5, 9 + width / 2 * 5]), admm_st_tv,
    #         alpha=0.9, width=width, hatch='*', edgecolor='black', label='ADMM+ST')

    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), grape_sur_tv, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+SUR+ALB')
    plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_sur_tv, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+SUR+ALB')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 + width / 2, 9 + width / 2]), admm_sur_tv, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+SUR+ALB')
    plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_sur_tv, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+SUR+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Rounding_improve_all_instances_new.png")


def draw_mt_improve():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_mt = [0.997, 0.834465772, 0.554418567, 0.805, 0.996, 0.994, 0.999, 0.992, 0.96]
    pgrape_mt = [0.755, 0.998]
    tr_mt = [0.997, 0.834465772, 0.554418567, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    admm_mt = [0.959, 0.81654493, 0.535016987, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    tr_st_mt = [0, 0.837013759, 0.621341738, 0.995, 0.593]
    admm_st_mt = [0, 0.635722779, 0.621341738, 0.869, 0.001]

    width = 0.15
    # plt.bar(np.array([1 - width * 2, 2 - width * 2, 3 - width * 2, 4 - width, 5 - width, 6 - width, 7 - width,
    #                   8 - width / 2 * 5, 9 - width / 2 * 5]), grape_mt, alpha=0.9, width=width,
    #         hatch='/', edgecolor='black', label='GRAPE+MT+ALB')
    # plt.bar(np.array([1 - width, 2 - width, 3 - width, 4, 5, 6, 7,
    #                   8 - width / 2 * 3, 9 - width / 2 * 3]), tr_mt, alpha=0.9, width=width,
    #         hatch='\\', edgecolor='black', label='TR+MT+ALB')
    # plt.bar(np.array([1, 2, 3, 4 + width, 5 + width, 6 + width, 7 + width,
    #                   8 - width / 2, 9 - width / 2]), admm_mt, alpha=0.9, width=width,
    #         hatch='+', edgecolor='black', label='ADMM+MT+ALB')
    # plt.bar(np.array([8 + width / 2, 9 + width / 2]), pgrape_mt, alpha=0.9, width=width,
    #         hatch='o', edgecolor='black', label='p-GRAPE+MT+ALB')
    # plt.bar(np.array([1 + width, 2 + width, 3 + width, 8 + width / 2 * 3, 9 + width / 2 * 3]), tr_st_mt, alpha=0.9,
    #         width=width, hatch='.', edgecolor='black', label='TR+STMT')
    # plt.bar(np.array([1 + width * 2, 2 + width * 2, 3 + width * 2, 8 + width / 2 * 5, 9 + width / 2 * 5]), admm_st_mt,
    #         alpha=0.9, width=width, hatch='*', edgecolor='black', label='ADMM+STMT')
    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), grape_mt, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+MT+ALB')
    plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_mt, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+MT+ALB')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 + width / 2, 9 + width / 2]), admm_mt, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+MT+ALB')
    plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_mt, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+MT+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Minup_time_improve_all_instances_new.png")


def draw_ms_improve():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_ms = [0.999, 0.857057924, 0.775408293, 0.83, 0.999, 0.999, 0.9990531, 0.997, 0.994]
    pgrape_ms = [0.986, 0.835]
    tr_ms = [0.998, 0.857482589, 0.773893823, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    admm_ms = [0.997, 0.840496008, 0.767835946, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]

    width = 0.15
    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), grape_ms, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+MS+ALB')
    plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_ms, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+MS+ALB')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 + width / 2, 9 + width / 2]), admm_ms, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+MS+ALB')
    plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_ms, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+MS+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Max_switching_improve_all_instances.png")


def draw_sur():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_sur_tv = [0.459, 0.745625955, 0.709303753, 0.670, 0.883, 0.972, 0.950, -0.062, -0.171]
    pgrape_sur_tv = [0.941, 0.452]
    tr_sur_tv = [0.455, 0.722694072, 0.6995211, 0.428, 0.883, 0.970, 0.951, 0.937, 0.438]
    admm_sur_tv = [0.519, 0.662051979, 0.676312881, 0.400, 0.913, 0.969, 0.952, 0.918, 0.715]

    tr_st_tv = [0, 0.830134194, 0.669190782, 0.987, 0.951]
    admm_st_tv = [0, 0.622728045, 0.669190782, 0.865, 0.001]

    width = 0.15
    # plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
    #                   8 - width / 2 * 3, 9 - width / 2 * 3]), grape_sur_tv, alpha=0.9, width=width,
    #         hatch='/', edgecolor='black', label='GRAPE+SUR')
    # plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_sur_tv, alpha=0.9, width=width,
    #         hatch='\\', edgecolor='black', label='TR+SUR')
    # plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
    #                   8 + width / 2, 9 + width / 2]), admm_sur_tv, alpha=0.9, width=width,
    #         hatch='+', edgecolor='black', label='ADMM+SUR')
    # plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_sur_tv, alpha=0.9, width=width,
    #         hatch='o', edgecolor='black', label='p-GRAPE+SUR')

    plt.bar(np.array([1 - width * 2, 2 - width * 2, 3 - width * 2, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 5, 9 - width / 2 * 5]), grape_sur_tv, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+SUR')
    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4, 5, 6, 7,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), tr_sur_tv, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+SUR')
    plt.bar(np.array([1, 2, 3, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 - width / 2, 9 - width / 2]), admm_sur_tv, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+SUR')
    plt.bar(np.array([8 + width / 2, 9 + width / 2]), pgrape_sur_tv, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+SUR')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 8 + width / 2 * 3, 9 + width / 2 * 3]), tr_st_tv, alpha=0.9,
            width=width, hatch='.', edgecolor='black', label='TR+ST')
    plt.bar(np.array([1 + width * 2, 2 + width * 2, 3 + width * 2, 8 + width / 2 * 5, 9 + width / 2 * 5]), admm_st_tv,
            alpha=0.9, width=width, hatch='*', edgecolor='black', label='ADMM+ST')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Rounding_all_instances_new.png")


def draw_mt():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_mt = [0.841, 0.653728554, 0.316892473, 0.757, 0.842, 0.461, 0.218, 0.291, 0.003]
    pgrape_mt = [0.400, 0.037]
    tr_mt = [0.841, 0.699677255, 0.382219312, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    admm_mt = [0.846, 0.649312043, 0.390978675, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    tr_st_mt = [0, 0.837013759, 0.621341738, 0.995, 0.593]
    admm_st_mt = [0, 0.635722779, 0.621341738, 0.869, 0.001]

    width = 0.15
    # plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
    #                   8 - width / 2 * 3, 9 - width / 2 * 3]), grape_mt, alpha=0.9, width=width,
    #         hatch='/', edgecolor='black', label='GRAPE+MT')
    # plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_mt, alpha=0.9, width=width,
    #         hatch='\\', edgecolor='black', label='TR+MT')
    # plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
    #                   8 + width / 2, 9 + width / 2]), admm_mt, alpha=0.9, width=width,
    #         hatch='+', edgecolor='black', label='ADMM+MT')
    # plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_mt, alpha=0.9, width=width,
    #         hatch='o', edgecolor='black', label='p-GRAPE+MT')
    plt.bar(np.array([1 - width * 2, 2 - width * 2, 3 - width * 2, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 5, 9 - width / 2 * 5]), grape_mt, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+MT')
    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4, 5, 6, 7,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), tr_mt, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+MT')
    plt.bar(np.array([1, 2, 3, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 - width / 2, 9 - width / 2]), admm_mt, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+MT')
    plt.bar(np.array([8 + width / 2, 9 + width / 2]), pgrape_mt, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+MT')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 8 + width / 2 * 3, 9 + width / 2 * 3]), tr_st_mt, alpha=0.9,
            width=width, hatch='.', edgecolor='black', label='TR+STMT')
    plt.bar(np.array([1 + width * 2, 2 + width * 2, 3 + width * 2, 8 + width / 2 * 5, 9 + width / 2 * 5]), admm_st_mt,
            alpha=0.9, width=width, hatch='*', edgecolor='black', label='ADMM+STMT')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Minup_time_instances_new.png")


def draw_ms():
    plt.figure(figsize=(15, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    grape_ms = [0.971, 0.854509937, 0.758544472, 0.830, 0.989, 0.675, 0.346, 0.230, 0.004]
    pgrape_ms = [0.974, 0.713]
    tr_ms = [0.96, 0.855868864, 0.747288281, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    admm_ms = [0.972, 0.829879395, 0.763374401, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]

    width = 0.15
    plt.bar(np.array([1 - width, 2 - width, 3 - width, 4 - width, 5 - width, 6 - width, 7 - width,
                      8 - width / 2 * 3, 9 - width / 2 * 3]), grape_ms, alpha=0.9, width=width,
            hatch='/', edgecolor='black', label='GRAPE+MS')
    plt.bar(np.array([1, 2, 3, 4, 5, 6, 7, 8 - width / 2, 9 - width / 2]), tr_ms, alpha=0.9, width=width,
            hatch='\\', edgecolor='black', label='TR+MS')
    plt.bar(np.array([1 + width, 2 + width, 3 + width, 4 + width, 5 + width, 6 + width, 7 + width,
                      8 + width / 2, 9 + width / 2]), admm_ms, alpha=0.9, width=width,
            hatch='+', edgecolor='black', label='ADMM+MS')
    plt.bar(np.array([8 + width / 2 * 3, 9 + width / 2 * 3]), pgrape_ms, alpha=0.9, width=width,
            hatch='o', edgecolor='black', label='p-GRAPE+MS')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figure_paper/Max_switching_all_instances.png")


def draw_grape_obj():
    grape_sur = [0.999, 0.859436046, 0.788711064, 0.830, 0.999, 0.999, 0.999, 0.050, 0.209]
    grape_mt = [0.841, 0.653728554, 0.316892473, 0.757, 0.842, 0.461, 0.218, 0.291, 0.003]
    grape_ms = [0.971, 0.854509937, 0.758544472, 0.830, 0.989, 0.675, 0.346, 0.230, 0.004]
    grape_sur_improve = [0.997, 0.858246985, 0.7870738, 0.824, 0.999, 0.999, 0.999544, 0.993, 0.551]
    grape_mt_improve = [0.997, 0.834465772, 0.554418567, 0.805, 0.996, 0.994, 0.999, 0.992, 0.96]
    grape_ms_improve = [0.999, 0.857057924, 0.775408293, 0.83, 0.999, 0.999, 0.9990531, 0.997, 0.994]

    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]

    plt.plot(instance, grape_sur, marker='o', linestyle='-', label='GRAPE+SUR')
    plt.plot(instance, grape_mt, marker='^', linestyle='-', label='GRAPE+MT')
    plt.plot(instance, grape_ms, marker='+', markersize='8', linestyle='-', label='GRAPE+MS')
    plt.plot(instance, grape_sur_improve, marker='o', linestyle='--', label='GRAPE+SUR+ALB')
    plt.plot(instance, grape_mt_improve, marker='^', linestyle='--', label='GRAPE+MT+ALB')
    plt.plot(instance, grape_ms_improve, marker='+', markersize='8', linestyle='--', label='GRAPE+MS+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/grape_obj_all_instances.png")


def draw_grape_obj_instance():
    grape = [0.999, 0.863597758, 0.791535344, 0.831, 1.000, 1.000, 1.000, 1.000, 1.000]
    grape_sur = [0.999, 0.841476808, 0.7836432, 0.830, 0.999, 0.999, 0.999, 0.050, 0.209]
    grape_mt = [0.841, 0.632912135, 0.3161038, 0.757, 0.842, 0.461, 0.218, 0.291, 0.003]
    grape_ms = [0.971, 0.836338827, 0.7525139, 0.830, 0.989, 0.675, 0.346, 0.230, 0.004]
    grape_sur_improve = [0.997, 0.840329552, 0.7816671, 0.824, 0.999, 0.999, 0.999544, 0.993, 0.551]
    grape_mt_improve = [0.997, 0.801350332, 0.5713352, 0.805, 0.996, 0.994, 0.999, 0.992, 0.96]
    grape_ms_improve = [0.999, 0.839518644, 0.7711258, 0.83, 0.999, 0.999, 0.9990531, 0.997, 0.994]

    all_methods = [grape, grape_sur, grape_mt, grape_ms, grape_sur_improve, grape_mt_improve, grape_ms_improve]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7])
    method_name = ["GRAPE", "GRAPE+SUR", "GRAPE+MT", "GRAPE+MS", "GRAPE+SUR+ALB", "GRAPE+MT+ALB", "GRAPE+MS+ALB"]

    plt.plot(methods, [1 - method[0] for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [1 - method[1] for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [1 - method[2] for method in all_methods], marker='+', markersize='8', linestyle='-',
             label="Energy6")
    plt.plot(methods, [1 - method[3] for method in all_methods], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods, [1 - method[4] for method in all_methods], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods, [1 - method[5] for method in all_methods], marker='+', markersize='8', linestyle='--',
             label="CNOT15")
    plt.plot(methods, [1 - method[6] for method in all_methods], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [1 - method[7] for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [1 - method[8] for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/grape_obj_per_instance_min.png")


def draw_grape_tv():
    grape_sur_tv = [54, 26.8, 38.8, 16, 116, 266, 491, 112, 380]
    grape_mt_tv = [4, 6, 6, 10, 22, 37, 53, 10, 48]
    grape_ms_tv = [10, 10, 10, 16, 39, 38, 39, 26, 208]
    grape_sur_improve_tv = [10, 17.2, 32.4, 9, 30, 262, 479, 16, 6]
    grape_mt_improve_tv = [4, 6, 6, 9, 23, 33, 28, 4, 18]
    grape_ms_improve_tv = [10, 9.2, 10, 16, 39, 39, 40, 18, 50]

    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2",
                     "CircuitLiH"]

    plt.plot(instance, grape_sur_tv, marker='o', linestyle='-', label='GRAPE+SUR')
    plt.plot(instance, grape_mt_tv, marker='^', linestyle='-', label='GRAPE+MT')
    plt.plot(instance, grape_ms_tv, marker='+', markersize='8', linestyle='-', label='GRAPE+MS')
    plt.plot(instance, grape_sur_improve_tv, marker='o', linestyle='--', label='GRAPE+SUR+ALB')
    plt.plot(instance, grape_mt_improve_tv, marker='^', linestyle='--', label='GRAPE+MT+ALB')
    plt.plot(instance, grape_ms_improve_tv, marker='+', markersize='8', linestyle='--', label='GRAPE+MS+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/grape_tv_all_instances.png")


def draw_grape_tv_instance():
    grape_tv = [0.999, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 2.744, 0.677]
    grape_sur_tv = [54, 26.8, 38.8, 16, 116, 266, 491, 112, 380]
    grape_mt_tv = [4, 6, 6, 10, 22, 37, 53, 10, 48]
    grape_ms_tv = [10, 10, 10, 16, 39, 38, 39, 26, 208]
    grape_sur_improve_tv = [10, 17.2, 32.4, 9, 30, 262, 479, 16, 6]
    grape_mt_improve_tv = [4, 6, 6, 9, 23, 33, 28, 4, 18]
    grape_ms_improve_tv = [10, 9.2, 10, 16, 39, 39, 40, 18, 50]

    all_methods = [grape_tv, grape_sur_tv, grape_mt_tv, grape_ms_tv, grape_sur_improve_tv, grape_mt_improve_tv,
                   grape_ms_improve_tv]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7])
    method_name = ["GRAPE", "GRAPE+SUR", "GRAPE+MT", "GRAPE+MS", "GRAPE+SUR+ALB", "GRAPE+MT+ALB", "GRAPE+MS+ALB"]

    plt.plot(methods, [np.log10(method[0]) for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [np.log10(method[1]) for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [np.log10(method[2]) for method in all_methods], marker='+', markersize='8', linestyle='-', label="Energy6")
    plt.plot(methods, [np.log10(method[3]) for method in all_methods], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods, [np.log10(method[4]) for method in all_methods], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods, [np.log10(method[5]) for method in all_methods], marker='+', markersize='8', linestyle='--', label="CNOT15")
    plt.plot(methods, [np.log10(method[6]) for method in all_methods], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [np.log10(method[7]) for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [np.log10(method[8]) for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize=8)
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/grape_tv_per_instance_log10.png")


def draw_pgrape_obj():
    pgrape_sur = [0.973, 0.832]
    pgrape_mt = [0.400, 0.037]
    pgrape_ms = [0.974, 0.713]
    pgrape_sur_improve = [0.997, 0.931]
    pgrape_mt_improve = [0.755, 0.998]
    pgrape_ms_improve = [0.986, 0.835]

    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2])
    instance_name = ["CircuitH2", "CircuitLiH"]

    plt.plot(instance, pgrape_sur, marker='o', linestyle='-', label='p-GRAPE+SUR')
    plt.plot(instance, pgrape_mt, marker='^', linestyle='-', label='p-GRAPE+MT')
    plt.plot(instance, pgrape_ms, marker='+', markersize='8', linestyle='-', label='p-GRAPE+MS')
    plt.plot(instance, pgrape_sur_improve, marker='o', linestyle='--', label='p-GRAPE+SUR+ALB')
    plt.plot(instance, pgrape_mt_improve, marker='^', linestyle='--', label='p-GRAPE+MT+ALB')
    plt.plot(instance, pgrape_ms_improve, marker='+', markersize='8', linestyle='--', label='p-GRAPE+MS+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/pgrape_obj_all_instances.png")


def draw_pgrape_obj_instance():
    pgrape = [1.000, 0.999]
    pgrape_sur = [0.973, 0.832]
    pgrape_mt = [0.400, 0.037]
    pgrape_ms = [0.974, 0.713]
    pgrape_sur_improve = [0.997, 0.931]
    pgrape_mt_improve = [0.755, 0.998]
    pgrape_ms_improve = [0.986, 0.835]

    all_methods = [pgrape, pgrape_sur, pgrape_mt, pgrape_ms, pgrape_sur_improve, pgrape_mt_improve, pgrape_ms_improve]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7])
    method_name = ["p-GRAPE", "p-GRAPE+SUR", "p-GRAPE+MT", "p-GRAPE+MS", "p-GRAPE+SUR+ALB", "p-GRAPE+MT+ALB",
                   "p-GRAPE+MS+ALB"]

    plt.plot(methods, [1 - method[0] for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [1 - method[1] for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/pgrape_obj_per_instance_min.png")


def draw_pgrape_tv():
    pgrape_sur_tv = [32, 380]
    pgrape_mt_tv = [8, 68]
    pgrape_ms_tv = [22, 290]
    pgrape_sur_improve_tv = [24, 378]
    pgrape_mt_improve_tv = [12, 6]
    pgrape_ms_improve_tv = [18, 286]

    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2])
    instance_name = ["CircuitH2", "CircuitLiH"]

    plt.plot(instance, pgrape_sur_tv, marker='o', linestyle='-', label='p-GRAPE+SUR')
    plt.plot(instance, pgrape_mt_tv, marker='^', linestyle='-', label='p-GRAPE+MT')
    plt.plot(instance, pgrape_ms_tv, marker='+', markersize='8', linestyle='-', label='p-GRAPE+MS')
    plt.plot(instance, pgrape_sur_improve_tv, marker='o', linestyle='--', label='p-GRAPE+SUR+ALB')
    plt.plot(instance, pgrape_mt_improve_tv, marker='^', linestyle='--', label='p-GRAPE+MT+ALB')
    plt.plot(instance, pgrape_ms_improve_tv, marker='+', markersize='8', linestyle='--', label='p-GRAPE+MS+ALB')

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/pgrape_tv_all_instances.png")


def draw_pgrape_tv_instance():
    pgrape_tv = [18.720, 53.976]
    pgrape_sur_tv = [32, 380]
    pgrape_mt_tv = [8, 68]
    pgrape_ms_tv = [22, 290]
    pgrape_sur_improve_tv = [24, 378]
    pgrape_mt_improve_tv = [12, 6]
    pgrape_ms_improve_tv = [18, 286]

    all_methods = [pgrape_tv, pgrape_sur_tv, pgrape_mt_tv, pgrape_ms_tv, pgrape_sur_improve_tv, pgrape_mt_improve_tv,
                   pgrape_ms_improve_tv]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7])
    method_name = ["p-GRAPE", "p-GRAPE+SUR", "p-GRAPE+MT", "p-GRAPE+MS", "p-GRAPE+SUR+ALB", "p-GRAPE+MT+ALB",
                   "p-GRAPE+MS+ALB"]

    plt.plot(methods, [np.log10(method[0]) for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [np.log10(method[1]) for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize=8)
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/pgrape_tv_per_instance_log10.png")


def draw_pgrape_obj_tv_instance_split():
    pgrape = [0.999, 0.863597758, 0.791535344, 0.831, 1.000, 1.000, 1.000, 1.000, 0.999]
    pgrape_sur = [0.999, 0.841476808, 0.7836432, 0.830, 0.999, 0.999, 0.999, 0.973, 0.832]
    pgrape_mt = [0.841, 0.632912135, 0.3161038, 0.757, 0.842, 0.461, 0.218, 0.400, 0.037]
    pgrape_ms = [0.971, 0.836338827, 0.7525139, 0.830, 0.989, 0.675, 0.346, 0.974, 0.713]
    pgrape_sur_improve = [0.997, 0.840329552, 0.7816671, 0.824, 0.999, 0.999, 0.999544, 0.997, 0.931]
    pgrape_mt_improve = [0.997, 0.801350332, 0.5713352, 0.805, 0.996, 0.994, 0.999, 0.755, 0.998]
    pgrape_ms_improve = [0.999, 0.839518644, 0.7711258, 0.83, 0.999, 0.999, 0.9990531, 0.986, 0.835]

    pgrape_tv = [0.999, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 18.720, 53.976]
    pgrape_sur_tv = [54, 26.8, 38.8, 16, 116, 266, 491, 32, 380]
    pgrape_mt_tv = [4, 6, 6, 10, 22, 37, 53, 8, 68]
    pgrape_ms_tv = []
    pgrape_sur_improve_tv = []
    pgrape_mt_improve_tv = []
    pgrape_ms_improve_tv = []
    pgrape_ms_tv = [10, 10, 10, 16, 39, 38, 39, 22, 290]
    pgrape_sur_improve_tv = [10, 17.2, 32.4, 9, 30, 262, 479, 24, 378]
    pgrape_mt_improve_tv = [4, 6, 6, 9, 23, 33, 28, 12, 6]
    pgrape_ms_improve_tv = [10, 9.2, 10, 16, 39, 39, 40, 18, 286]

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [pgrape, pgrape_sur, pgrape_sur_improve]
    methods_mt = [pgrape, pgrape_mt, pgrape_mt_improve]
    methods_ms = [pgrape, pgrape_ms, pgrape_ms_improve]

    methods_sur_tv = [pgrape_tv, pgrape_sur_tv, pgrape_sur_improve_tv]
    methods_mt_tv = [pgrape_tv, pgrape_mt_tv, pgrape_mt_improve_tv]
    methods_ms_tv = [pgrape_tv, pgrape_ms_tv, pgrape_ms_improve_tv]
    # all_methods = [pgrape, pgrape_sur, pgrape_mt, pgrape_ms, pgrape_sur_improve, pgrape_mt_improve, pgrape_ms_improve,
    #                pgrape_st, pgrape_stmt]
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10)

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3)

    plt.savefig("../figure_paper/pgrape_obj_split.png")

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10)
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3)

    plt.savefig("../figure_paper/pgrape_tv_split.png")


def draw_pgrape_selected():
    pgrape = [0.999, 0.863597758, 0.791535344, 0.831, 1.000, 1.000, 1.000, 1.000, 0.999]
    pgrape_sur = [0.999, 0.841476808, 0.7836432, 0.830, 0.999, 0.999, 0.999, 0.973, 0.832]
    pgrape_mt = [0.841, 0.632912135, 0.3161038, 0.757, 0.842, 0.461, 0.218, 0.400, 0.037]
    pgrape_ms = [0.971, 0.836338827, 0.7525139, 0.830, 0.989, 0.675, 0.346, 0.974, 0.713]
    pgrape_sur_improve = [0.997, 0.840329552, 0.7816671, 0.824, 0.999, 0.999, 0.999544, 0.997, 0.931]
    pgrape_mt_improve = [0.997, 0.801350332, 0.5713352, 0.805, 0.996, 0.994, 0.999, 0.755, 0.998]
    pgrape_ms_improve = [0.999, 0.839518644, 0.7711258, 0.83, 0.999, 0.999, 0.9990531, 0.986, 0.835]

    pgrape_tv = [0.999, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 18.720, 53.976]
    pgrape_sur_tv = [54, 26.8, 38.8, 16, 116, 266, 491, 32, 380]
    pgrape_mt_tv = [4, 6, 6, 10, 22, 37, 53, 8, 68]
    pgrape_ms_tv = []
    pgrape_sur_improve_tv = []
    pgrape_mt_improve_tv = []
    pgrape_ms_improve_tv = []
    pgrape_ms_tv = [10, 10, 10, 16, 39, 38, 39, 22, 290]
    pgrape_sur_improve_tv = [10, 17.2, 32.4, 9, 30, 262, 479, 24, 378]
    pgrape_mt_improve_tv = [4, 6, 6, 9, 23, 33, 28, 12, 6]
    pgrape_ms_improve_tv = [10, 9.2, 10, 16, 39, 39, 40, 18, 286]

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [pgrape, pgrape_sur, pgrape_sur_improve]
    methods_mt = [pgrape, pgrape_mt, pgrape_mt_improve]
    methods_ms = [pgrape, pgrape_ms, pgrape_ms_improve]

    methods_sur_tv = [pgrape_tv, pgrape_sur_tv, pgrape_sur_improve_tv]
    methods_mt_tv = [pgrape_tv, pgrape_mt_tv, pgrape_mt_improve_tv]
    methods_ms_tv = [pgrape_tv, pgrape_ms_tv, pgrape_ms_improve_tv]

    select = [2, 6, 8]
    # all_methods = [pgrape, pgrape_sur, pgrape_mt, pgrape_ms, pgrape_sur_improve, pgrape_mt_improve, pgrape_ms_improve,
    #                pgrape_st, pgrape_stmt]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10)

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/pgrape_obj_select.png")

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10)
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/pgrape_tv_select.png")


def draw_admm_obj():
    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]

    admm_sur = [0.999, 0.829315975, 0.7788632, 0.810, 0.999, 0.997, 0.999, 0.994, 0.967]
    admm_mt = [0.846, 0.636487265, 0.3936237, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    admm_ms = [0.972, 0.805007335, 0.7596211, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]
    admm_sur_improve = [0.997, 0.833747845, 0.7808066, 0.824, 0.999, 0.999, 0.999, 0.994, 0.967]
    admm_mt_improve = [0.959, 0.781849407, 0.5482583, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    admm_ms_improve = [0.997, 0.815520921, 0.7639767, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]
    admm_st = [0, 0.525196884, 0.6789286, 0.869, 0.001]
    admm_stmt = [0, 0.521655792, 0.6242092, 0.869, 0.001]

    plt.plot(instance, admm_sur, marker='o', linestyle='-', label='ADMM+SUR')
    plt.plot(instance, admm_mt, marker='^', linestyle='-', label='ADMM+MT')
    plt.plot(instance, admm_ms, marker='+', markersize='8', linestyle='-', label='ADMM+MS')
    plt.plot(instance, admm_sur_improve, marker='o', linestyle='--', label='ADMM+SUR+ALB')
    plt.plot(instance, admm_mt_improve, marker='^', linestyle='--', label='ADMM+MT+ALB')
    plt.plot(instance, admm_ms_improve, marker='+', markersize='8', linestyle='--', label='ADMM+MS+ALB')
    line1 = plt.plot(instance[0:3], admm_st[0:3], marker='o', linestyle='dotted', label='ADMM+ST')
    plt.plot(instance[7:], admm_st[3:], marker='o', linestyle='dotted', color=line1[0].get_color())
    line2 = plt.plot(instance[0:3], admm_stmt[0:3], marker='^', linestyle='dotted', label='ADMM+STMT')
    plt.plot(instance[7:], admm_stmt[3:], marker='^', linestyle='dotted', color=line2[0].get_color())

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/admm_obj_all_instances.png")


def draw_admm_obj_instance():
    admm = [0.999, 0.853660608, 0.785477467, 0.809, 1.000, 1.000, 1.000, 1.000, 0.999]
    admm_sur = [0.999, 0.829315975, 0.7788632, 0.810, 0.999, 0.997, 0.999, 0.994, 0.967]
    admm_mt = [0.846, 0.636487265, 0.3936237, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    admm_ms = [0.972, 0.805007335, 0.7596211, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]
    admm_sur_improve = [0.997, 0.833747845, 0.7808066, 0.824, 0.999, 0.999, 0.999, 0.994, 0.967]
    admm_mt_improve = [0.959, 0.781849407, 0.5482583, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    admm_ms_improve = [0.997, 0.815520921, 0.7639767, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]
    admm_st = [0, 0.525196884, 0.6789286, None, None, None, None, 0.869, 0.001]
    admm_stmt = [0, 0.521655792, 0.6242092, None, None, None, None, 0.869, 0.001]

    all_methods = [admm, admm_sur, admm_mt, admm_ms, admm_sur_improve, admm_mt_improve, admm_ms_improve, admm_st,
                   admm_stmt]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    method_name = ["ADMM", "ADMM+SUR", "ADMM+MT", "ADMM+MS", "ADMM+SUR+ALB", "ADMM+MT+ALB", "ADMM+MS+ALB",
                   "ADMM+ST", "ADMM+STMT"]

    plt.plot(methods, [1 - method[0] for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [1 - method[1] for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [1 - method[2] for method in all_methods], marker='+', markersize='8', linestyle='-',
             label="Energy6")
    plt.plot(methods[:-2], [1 - method[3] for method in all_methods[:-2]], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods[:-2], [1 - method[4] for method in all_methods[:-2]], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods[:-2], [1 - method[5] for method in all_methods[:-2]], marker='+', markersize='8', linestyle='--',
             label="CNOT15")
    plt.plot(methods[:-2], [1 - method[6] for method in all_methods[:-2]], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [1 - method[7] for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [1 - method[8] for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/admm_obj_per_instance_min.png")


def draw_admm_instance_split():
    admm = [0.999, 0.853660608, 0.785477467, 0.809, 1.000, 1.000, 1.000, 1.000, 0.999]
    admm_sur = [0.999, 0.829315975, 0.7788632, 0.810, 0.999, 0.997, 0.999, 0.994, 0.967]
    admm_mt = [0.846, 0.636487265, 0.3936237, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    admm_ms = [0.972, 0.805007335, 0.7596211, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]
    admm_sur_improve = [0.997, 0.833747845, 0.7808066, 0.824, 0.999, 0.999, 0.999, 0.994, 0.967]
    admm_mt_improve = [0.959, 0.781849407, 0.5482583, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    admm_ms_improve = [0.997, 0.815520921, 0.7639767, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]
    admm_st = [0, 0.525196884, 0.6789286, None, None, None, None, 0.869, 0.001]
    admm_stmt = [0, 0.521655792, 0.6242092, None, None, None, None, 0.869, 0.001]

    admm_tv = [0.567, 4.114, 4.508, 9.419, 15.194, 24.348, 23.481, 8.421, 48.720]
    admm_sur_tv = [48, 44.8, 52.8, 41, 86, 279, 467, 76, 252]
    admm_mt_tv = [6, 6, 6, 7, 15, 27, 47, 8, 48]
    admm_ms_tv = [10, 10, 10, 32, 32, 39, 39, 22, 148]
    admm_sur_improve_tv = [4, 14.4, 50.0, 9, 20, 263, 441, 76, 252]
    admm_mt_improve_tv = [6, 5.6, 6, 9, 15, 30, 48, 10, 48]
    admm_ms_improve_tv = [8, 8.4, 10, 16, 36, 40, 40, 22, 158]
    admm_st_tv = [0, 4, 6, None, None, None, None, 4, 0]
    admm_stmt_tv = [0, 4, 6, None, None, None, None, 4, 0]

    fig = plt.figure(figsize=(11, 16), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.35, left=0.05, right=0.95, top=0.98, bottom=0.04)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [admm, admm_sur, admm_sur_improve]
    methods_mt = [admm, admm_mt, admm_mt_improve]
    methods_ms = [admm, admm_ms, admm_ms_improve]
    methods_st = [admm, admm_st]
    methods_stmt = [admm, admm_stmt]

    methods_sur_tv = [admm_tv, admm_sur_tv, admm_sur_improve_tv]
    methods_mt_tv = [admm_tv, admm_mt_tv, admm_mt_improve_tv]
    methods_ms_tv = [admm_tv, admm_ms_tv, admm_ms_improve_tv]
    methods_st_tv = [admm_tv, admm_st_tv]
    methods_stmt_tv = [admm_tv, admm_stmt_tv]
    # all_methods = [admm, admm_sur, admm_mt, admm_ms, admm_sur_improve, admm_mt_improve, admm_ms_improve, admm_st,
    #                admm_stmt]
    for i in range(18):
        ax = fig.add_subplot(6, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])
        else:
            ax.set_title(instance_name[i - 9])

        if i == 0:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--', label='STMT')
        if 1 <= i <= 8:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--')
                plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--')

        if i > 8:
            plt.plot(models, [method[i - 9] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[i - 9] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[i - 9] for method in methods_ms_tv], marker='+', markersize=10)
            if i in [9, 10, 11, 16, 17]:
                plt.plot(models[0:2], [method[i - 9] for method in methods_st_tv], marker='o', linestyle='--')
                plt.plot(models[0:2], [method[i - 9] for method in methods_stmt_tv], marker='^', linestyle='--')

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=0)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0, ncol=5)
        # , bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0, ncol=5)

    # fig.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    # fig.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/admm_split.png")


def draw_admm_selected():
    admm = [0.999, 0.853660608, 0.785477467, 0.809, 1.000, 1.000, 1.000, 1.000, 0.999]
    admm_sur = [0.999, 0.829315975, 0.7788632, 0.810, 0.999, 0.997, 0.999, 0.994, 0.967]
    admm_mt = [0.846, 0.636487265, 0.3936237, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    admm_ms = [0.972, 0.805007335, 0.7596211, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]
    admm_sur_improve = [0.997, 0.833747845, 0.7808066, 0.824, 0.999, 0.999, 0.999, 0.994, 0.967]
    admm_mt_improve = [0.959, 0.781849407, 0.5482583, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    admm_ms_improve = [0.997, 0.815520921, 0.7639767, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]
    admm_st = [0, 0.525196884, 0.6789286, None, None, None, None, 0.869, 0.001]
    admm_stmt = [0, 0.521655792, 0.6242092, None, None, None, None, 0.869, 0.001]

    admm_tv = [0.567, 4.114, 4.508, 9.419, 15.194, 24.348, 23.481, 8.421, 48.720]
    admm_sur_tv = [48, 44.8, 52.8, 41, 86, 279, 467, 76, 252]
    admm_mt_tv = [6, 6, 6, 7, 15, 27, 47, 8, 48]
    admm_ms_tv = [10, 10, 10, 32, 32, 39, 39, 22, 148]
    admm_sur_improve_tv = [4, 14.4, 50.0, 9, 20, 263, 441, 76, 252]
    admm_mt_improve_tv = [6, 5.6, 6, 9, 15, 30, 48, 10, 48]
    admm_ms_improve_tv = [8, 8.4, 10, 16, 36, 40, 40, 22, 158]
    admm_st_tv = [0, 4, 6, None, None, None, None, 4, 0]
    admm_stmt_tv = [0, 4, 6, None, None, None, None, 4, 0]

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [admm, admm_sur, admm_sur_improve]
    methods_mt = [admm, admm_mt, admm_mt_improve]
    methods_ms = [admm, admm_ms, admm_ms_improve]
    methods_st = [admm, admm_st]
    methods_stmt = [admm, admm_stmt]

    methods_sur_tv = [admm_tv, admm_sur_tv, admm_sur_improve_tv]
    methods_mt_tv = [admm_tv, admm_mt_tv, admm_mt_improve_tv]
    methods_ms_tv = [admm_tv, admm_ms_tv, admm_ms_improve_tv]
    methods_st_tv = [admm_tv, admm_st_tv]
    methods_stmt_tv = [admm_tv, admm_stmt_tv]

    select = [2, 6, 8]
    # all_methods = [pgrape, pgrape_sur, pgrape_mt, pgrape_ms, pgrape_sur_improve, pgrape_mt_improve, pgrape_ms_improve,
    #                pgrape_st, pgrape_stmt]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10)

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/admm_obj_select.png")

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10)
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/admm_tv_select.png")


def draw_admm_obj_tv_instance_split():
    admm = [0.999, 0.853660608, 0.785477467, 0.809, 1.000, 1.000, 1.000, 1.000, 0.999]
    admm_sur = [0.999, 0.829315975, 0.7788632, 0.810, 0.999, 0.997, 0.999, 0.994, 0.967]
    admm_mt = [0.846, 0.636487265, 0.3936237, 0.475, 0.916, 0.824, 0.483, 0.937, 0.342]
    admm_ms = [0.972, 0.805007335, 0.7596211, 0.809, 0.994, 0.786, 0.381, 0.992, 0.443]
    admm_sur_improve = [0.997, 0.833747845, 0.7808066, 0.824, 0.999, 0.999, 0.999, 0.994, 0.967]
    admm_mt_improve = [0.959, 0.781849407, 0.5482583, 0.805, 0.994, 0.999, 0.999, 0.946, 0.645]
    admm_ms_improve = [0.997, 0.815520921, 0.7639767, 0.828, 0.999, 0.998, 0.9992551, 0.992, 0.979]
    admm_st = [0, 0.525196884, 0.6789286, None, None, None, None, 0.869, 0.001]
    admm_stmt = [0, 0.521655792, 0.6242092, None, None, None, None, 0.869, 0.001]

    admm_tv = [0.567, 4.114, 4.508, 9.419, 15.194, 24.348, 23.481, 8.421, 48.720]
    admm_sur_tv = [48, 44.8, 52.8, 41, 86, 279, 467, 76, 252]
    admm_mt_tv = [6, 6, 6, 7, 15, 27, 47, 8, 48]
    admm_ms_tv = [10, 10, 10, 32, 32, 39, 39, 22, 148]
    admm_sur_improve_tv = [4, 14.4, 50.0, 9, 20, 263, 441, 76, 252]
    admm_mt_improve_tv = [6, 5.6, 6, 9, 15, 30, 48, 10, 48]
    admm_ms_improve_tv = [8, 8.4, 10, 16, 36, 40, 40, 22, 158]
    admm_st_tv = [0, 4, 6, None, None, None, None, 4, 0]
    admm_stmt_tv = [0, 4, 6, None, None, None, None, 4, 0]

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [admm, admm_sur, admm_sur_improve]
    methods_mt = [admm, admm_mt, admm_mt_improve]
    methods_ms = [admm, admm_ms, admm_ms_improve]
    methods_st = [admm, admm_st]
    methods_stmt = [admm, admm_stmt]

    methods_sur_tv = [admm_tv, admm_sur_tv, admm_sur_improve_tv]
    methods_mt_tv = [admm_tv, admm_mt_tv, admm_mt_improve_tv]
    methods_ms_tv = [admm_tv, admm_ms_tv, admm_ms_improve_tv]
    methods_st_tv = [admm_tv, admm_st_tv]
    methods_stmt_tv = [admm_tv, admm_stmt_tv]
    # all_methods = [admm, admm_sur, admm_mt, admm_ms, admm_sur_improve, admm_mt_improve, admm_ms_improve, admm_st,
    #                admm_stmt]
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--', label='STMT')
        else:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--')
                plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--')

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=5)

    plt.savefig("../figure_paper/admm_obj_split.png")

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [method[i] for method in methods_st_tv], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [method[i] for method in methods_stmt_tv], marker='^', linestyle='--', label='STMT')
        else:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [method[i] for method in methods_st_tv], marker='o', linestyle='--')
                plt.plot(models[0:2], [method[i] for method in methods_stmt_tv], marker='^', linestyle='--')
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=5)

    plt.savefig("../figure_paper/admm_tv_split.png")


def draw_admm_tv():
    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]

    admm_sur_tv = [48, 44.8, 52.8, 41, 86, 279, 467, 56, 252]
    admm_mt_tv = [6, 6, 6, 7, 15, 27, 47, 8, 48]
    admm_ms_tv = [10, 10, 10, 32, 32, 39, 39, 22, 148]
    admm_sur_improve_tv = [4, 14.4, 50.0, 9, 20, 263, 441, 76, 252]
    admm_mt_improve_tv = [6, 5.6, 6, 9, 15, 30, 48, 10, 48]
    admm_ms_improve_tv = [8, 8.4, 10, 16, 36, 40, 40, 22, 158]
    admm_st_tv = [0, 4, 6, 4, 0]
    admm_stmt_tv = [0, 4, 6, 4, 0]

    plt.plot(instance, admm_sur_tv, marker='o', linestyle='-', label='ADMM+SUR')
    plt.plot(instance, admm_mt_tv, marker='^', linestyle='-', label='ADMM+MT')
    plt.plot(instance, admm_ms_tv, marker='+', markersize='8', linestyle='-', label='ADMM+MS')
    plt.plot(instance, admm_sur_improve_tv, marker='o', linestyle='--', label='ADMM+SUR+ALB')
    plt.plot(instance, admm_mt_improve_tv, marker='^', linestyle='--', label='ADMM+MT+ALB')
    plt.plot(instance, admm_ms_improve_tv, marker='+', markersize='8', linestyle='--', label='ADMM+MS+ALB')
    line1 = plt.plot(instance[0:3], admm_st_tv[0:3], marker='o', linestyle='dotted', label='ADMM+ST')
    plt.plot(instance[7:], admm_st_tv[3:], marker='o', linestyle='dotted', color=line1[0].get_color())
    line2 = plt.plot(instance[0:3], admm_stmt_tv[0:3], marker='^', linestyle='dotted', label='ADMM+STMT')
    plt.plot(instance[7:], admm_stmt_tv[3:], marker='^', linestyle='dotted', color=line2[0].get_color())

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/admm_tv_all_instances.png")


def draw_admm_tv_instance():
    admm_tv = [0.567, 4.114, 4.508, 9.419, 15.194, 24.348, 23.481, 8.421, 48.720]
    admm_sur_tv = [48, 44.8, 52.8, 41, 86, 279, 467, 56, 252]
    admm_mt_tv = [6, 6, 6, 7, 15, 27, 47, 8, 48]
    admm_ms_tv = [10, 10, 10, 32, 32, 39, 39, 22, 148]
    admm_sur_improve_tv = [4, 14.4, 50.0, 9, 20, 263, 441, 76, 252]
    admm_mt_improve_tv = [6, 5.6, 6, 9, 15, 30, 48, 10, 48]
    admm_ms_improve_tv = [8, 8.4, 10, 16, 36, 40, 40, 22, 158]
    admm_st_tv = [0, 4, 6, None, None, None, None, 4, 0]
    admm_stmt_tv = [0, 4, 6, None, None, None, None, 4, 0]

    all_methods = [admm_tv, admm_sur_tv, admm_mt_tv, admm_ms_tv, admm_sur_improve_tv, admm_mt_improve_tv,
                   admm_ms_improve_tv, admm_st_tv, admm_stmt_tv]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    method_name = ["ADMM", "ADMM+SUR", "ADMM+MT", "ADMM+MS", "ADMM+SUR+ALB", "ADMM+MT+ALB", "ADMM+MS+ALB",
                   "ADMM+ST", "ADMM+STMT"]

    plt.plot(methods, [np.log10(method[0]) for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [np.log10(method[1]) for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [np.log10(method[2]) for method in all_methods], marker='+', markersize='8', linestyle='-',
             label="Energy6")
    plt.plot(methods[:-2], [np.log10(method[3]) for method in all_methods[:-2]], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods[:-2], [np.log10(method[4]) for method in all_methods[:-2]], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods[:-2], [np.log10(method[5]) for method in all_methods[:-2]], marker='+', markersize='8', linestyle='--',
             label="CNOT15")
    plt.plot(methods[:-2], [np.log10(method[6]) for method in all_methods[:-2]], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [np.log10(method[7]) for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [np.log10(method[8]) for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize=8)
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/admm_tv_per_instance_log10.png")


def draw_tr_obj():
    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]

    tr_sur = [0.995, 0.860285375, 0.788751995, 0.668, 0.998, 0.998, 0.999, 0.973, 0.818]
    tr_mt = [0.841, 0.699677255, 0.382219312, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    tr_ms = [0.96, 0.855868864, 0.747288281, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    tr_sur_improve = [0.995, 0.856463394, 0.788670132, 0.794, 0.998, 1.000, 0.999, 0.987, 0.917]
    tr_mt_improve = [0.997, 0.834465772, 0.554418567, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    tr_ms_improve = [0.998, 0.857482589, 0.773893823, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    tr_st = [0, 0.855614065, 0.681470263, 0.995, 0.999]
    tr_stmt = [0, 0.837013759, 0.621341738, 0.995, 0.593]

    plt.plot(instance, tr_sur, marker='o', linestyle='-', label='TR+SUR')
    plt.plot(instance, tr_mt, marker='^', linestyle='-', label='TR+MT')
    plt.plot(instance, tr_ms, marker='+', markersize='8', linestyle='-', label='TR+MS')
    plt.plot(instance, tr_sur_improve, marker='o', linestyle='--', label='TR+SUR+ALB')
    plt.plot(instance, tr_mt_improve, marker='^', linestyle='--', label='TR+MT+ALB')
    plt.plot(instance, tr_ms_improve, marker='+', markersize='8', linestyle='--', label='TR+MS+ALB')
    line1 = plt.plot(instance[0:3], tr_st[0:3], marker='o', linestyle='dotted', label='TR+ST')
    plt.plot(instance[7:], tr_st[3:], marker='o', linestyle='dotted', color=line1[0].get_color())
    line2 = plt.plot(instance[0:3], tr_stmt[0:3], marker='^', linestyle='dotted', label='TR+STMT')
    plt.plot(instance[7:], tr_stmt[3:], marker='^', linestyle='dotted', color=line2[0].get_color())

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/tr_obj_all_instances.png")


def draw_tr_obj_instance():
    tr = [0.999, 0.862748429, 0.791003234, 0.876, 1.000, 1.000, 1.000, 1.000, 0.999]
    tr_sur = [0.995, 0.841779059, 0.7838377, 0.668, 0.998, 0.998, 0.999, 0.973, 0.818]
    tr_mt = [0.841, 0.682912135, 0.3832050, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    tr_ms = [0.96, 0.83739821, 0.7419845, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    tr_sur_improve = [0.995, 0.838262615, 0.7837403, 0.794, 0.998, 1.000, 0.999, 0.987, 0.917]
    tr_mt_improve = [0.997, 0.801350332, 0.5713352, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    tr_ms_improve = [0.998, 0.840041622, 0.7699510, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    tr_st = [0, 0.838716809, 0.6789286, None, None, None, None, 0.995, 0.999]
    tr_stmt = [0, 0.803798946, 0.6242092, None, None, None, None, 0.995, 0.593]

    all_methods = [tr, tr_sur, tr_mt, tr_ms, tr_sur_improve, tr_mt_improve, tr_ms_improve, tr_st, tr_stmt]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    method_name = ["TR", "TR+SUR", "TR+MT", "TR+MS", "TR+SUR+ALB", "TR+MT+ALB", "TR+MS+ALB", "TR+ST", "TR+STMT"]

    plt.plot(methods, [1 - method[0] for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [1 - method[1] for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [1 - method[2] for method in all_methods], marker='+', markersize='8', linestyle='-',
             label="Energy6")
    plt.plot(methods[:-2], [1 - method[3] for method in all_methods[:-2]], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods[:-2], [1 - method[4] for method in all_methods[:-2]], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods[:-2], [1 - method[5] for method in all_methods[:-2]], marker='+', markersize='8', linestyle='--',
             label="CNOT15")
    plt.plot(methods[:-2], [1 - method[6] for method in all_methods[:-2]], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [1 - method[7] for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [1 - method[8] for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/tr_obj_per_instance_min.png")


def draw_tr_instance_split():
    tr = [0.999, 0.862748429, 0.791003234, 0.876, 1.000, 1.000, 1.000, 1.000, 0.999]
    tr_sur = [0.995, 0.841779059, 0.7838377, 0.668, 0.998, 0.998, 0.999, 0.973, 0.818]
    tr_mt = [0.841, 0.682912135, 0.3832050, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    tr_ms = [0.96, 0.83739821, 0.7419845, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    tr_sur_improve = [0.995, 0.838262615, 0.7837403, 0.794, 0.998, 1.000, 0.999, 0.987, 0.917]
    tr_mt_improve = [0.997, 0.801350332, 0.5713352, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    tr_ms_improve = [0.998, 0.840041622, 0.7699510, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    tr_st = [0, 0.838716809, 0.6789286, None, None, None, None, 0.995, 0.999]
    tr_stmt = [0, 0.803798946, 0.6242092, None, None, None, None, 0.995, 0.593]

    tr_tv = [0.523, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 2.744, 0.677]
    tr_sur_tv = [54, 32.4, 43.6, 24, 116, 276, 480, 36, 280]
    tr_mt_tv = [6, 6, 6, 6, 21, 36, 51, 8, 72]
    tr_ms_tv = [10, 10, 10, 15, 38, 40, 39, 24, 288]
    tr_sur_improve_tv = [8, 14, 40.4, 7, 24, 256, 471, 32, 378]
    tr_mt_improve_tv = [4, 6, 6, 10, 16, 34, 49, 2, 70]
    tr_ms_improve_tv = [10, 10, 10, 20, 38, 39, 40, 22, 290]
    tr_st_tv = [2, 6, 6, None, None, None, None, 8, 48]
    tr_stmt_tv = [2, 6, 6, None, None, None, None, 8, 64]

    fig = plt.figure(figsize=(11, 16), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.35, left=0.05, right=0.95, top=0.98, bottom=0.04)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [tr, tr_sur, tr_sur_improve]
    methods_mt = [tr, tr_mt, tr_mt_improve]
    methods_ms = [tr, tr_ms, tr_ms_improve]
    methods_st = [tr, tr_st]
    methods_stmt = [tr, tr_stmt]

    methods_sur_tv = [tr_tv, tr_sur_tv, tr_sur_improve_tv]
    methods_mt_tv = [tr_tv, tr_mt_tv, tr_mt_improve_tv]
    methods_ms_tv = [tr_tv, tr_ms_tv, tr_ms_improve_tv]
    methods_st_tv = [tr_tv, tr_st_tv]
    methods_stmt_tv = [tr_tv, tr_stmt_tv]
    # all_methods = [tr, tr_sur, tr_mt, tr_ms, tr_sur_improve, tr_mt_improve, tr_ms_improve, tr_st,
    #                tr_stmt]
    for i in range(18):
        ax = fig.add_subplot(6, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])
        else:
            ax.set_title(instance_name[i - 9])

        if i == 0:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--', label='STMT')
        if 1 <= i <= 8:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--')
                plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--')

        if i > 8:
            plt.plot(models, [method[i - 9] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[i - 9] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[i - 9] for method in methods_ms_tv], marker='+', markersize=10)
            if i in [9, 10, 11, 16, 17]:
                plt.plot(models[0:2], [method[i - 9] for method in methods_st_tv], marker='o', linestyle='--')
                plt.plot(models[0:2], [method[i - 9] for method in methods_stmt_tv], marker='^', linestyle='--')

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=0)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=5)
        # , bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0, ncol=5)

    # fig.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    # fig.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/tr_split.png")


def draw_tr_obj_tv_instance_split():
    tr = [0.999, 0.862748429, 0.791003234, 0.876, 1.000, 1.000, 1.000, 1.000, 0.999]
    tr_sur = [0.995, 0.841779059, 0.7838377, 0.668, 0.998, 0.998, 0.999, 0.973, 0.818]
    tr_mt = [0.841, 0.682912135, 0.3832050, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    tr_ms = [0.96, 0.83739821, 0.7419845, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    tr_sur_improve = [0.995, 0.838262615, 0.7837403, 0.794, 0.998, 1.000, 0.999, 0.987, 0.917]
    tr_mt_improve = [0.997, 0.801350332, 0.5713352, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    tr_ms_improve = [0.998, 0.840041622, 0.7699510, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    tr_st = [0, 0.838716809, 0.6789286, None, None, None, None, 0.995, 0.999]
    tr_stmt = [0, 0.803798946, 0.6242092, None, None, None, None, 0.995, 0.593]

    tr_tv = [0.523, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 2.744, 0.677]
    tr_sur_tv = [54, 32.4, 43.6, 24, 116, 276, 480, 36, 380]
    tr_mt_tv = [6, 6, 6, 6, 21, 36, 51, 8, 72]
    tr_ms_tv = [10, 10, 10, 15, 38, 40, 39, 24, 288]
    tr_sur_improve_tv = [8, 14, 40.4, 7, 24, 256, 471, 32, 378]
    tr_mt_improve_tv = [4, 6, 6, 10, 16, 34, 49, 2, 70]
    tr_ms_improve_tv = [10, 10, 10, 20, 38, 39, 40, 22, 290]
    tr_st_tv = [2, 6, 6, None, None, None, None, 8, 48]
    tr_stmt_tv = [2, 6, 6, None, None, None, None, 8, 64]

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [tr, tr_sur, tr_sur_improve]
    methods_mt = [tr, tr_mt, tr_mt_improve]
    methods_ms = [tr, tr_ms, tr_ms_improve]
    methods_st = [tr, tr_st]
    methods_stmt = [tr, tr_stmt]

    methods_sur_tv = [tr_tv, tr_sur_tv, tr_sur_improve_tv]
    methods_mt_tv = [tr_tv, tr_mt_tv, tr_mt_improve_tv]
    methods_ms_tv = [tr_tv, tr_ms_tv, tr_ms_improve_tv]
    methods_st_tv = [tr_tv, tr_st_tv]
    methods_stmt_tv = [tr_tv, tr_stmt_tv]
    # all_methods = [tr, tr_sur, tr_mt, tr_ms, tr_sur_improve, tr_mt_improve, tr_ms_improve, tr_st,
    #                tr_stmt]
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--', label='STMT')
        else:
            plt.plot(models, [1 - method[i] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[i] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[i] for method in methods_ms], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [1 - method[i] for method in methods_st], marker='o', linestyle='--')
                plt.plot(models[0:2], [1 - method[i] for method in methods_stmt], marker='^', linestyle='--')

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=5)

    plt.savefig("../figure_paper/tr_obj_split.png")

    fig = plt.figure(figsize=(9, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.08)

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[i])

        if i == 0:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
            plt.plot(models[0:2], [method[i] for method in methods_st_tv], marker='o', linestyle='--', label='ST')
            plt.plot(models[0:2], [method[i] for method in methods_stmt_tv], marker='^', linestyle='--', label='STMT')
        else:
            plt.plot(models, [method[i] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[i] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[i] for method in methods_ms_tv], marker='+', markersize=10)
            if i in [0, 1, 2, 7, 8]:
                plt.plot(models[0:2], [method[i] for method in methods_st_tv], marker='o', linestyle='--')
                plt.plot(models[0:2], [method[i] for method in methods_stmt_tv], marker='^', linestyle='--')
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.25, 0, 0.5, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=5)

    plt.savefig("../figure_paper/tr_tv_split.png")


def draw_tr_tv():
    plt.figure(figsize=(8, 6), dpi=300)
    instance = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]

    tr_sur_tv = [54, 32.4, 43.6, 24, 116, 276, 480, 36, 280]
    tr_mt_tv = [6, 6, 6, 6, 21, 36, 51, 8, 72]
    tr_ms_tv = [10, 10, 10, 15, 38, 40, 39, 24, 288]
    tr_sur_improve_tv = [8, 14, 40.4, 7, 24, 256, 471, 32, 378]
    tr_mt_improve_tv = [4, 6, 6, 10, 16, 34, 49, 2, 70]
    tr_ms_improve_tv = [10, 10, 10, 20, 38, 39, 40, 22, 290]
    tr_st_tv = [2, 6, 6, 8, 48]
    tr_stmt_tv = [2, 6, 6, 8, 64]

    plt.plot(instance, tr_sur_tv, marker='o', linestyle='-', label='TR+SUR')
    plt.plot(instance, tr_mt_tv, marker='^', linestyle='-', label='TR+MT')
    plt.plot(instance, tr_ms_tv, marker='+', markersize='8', linestyle='-', label='TR+MS')
    plt.plot(instance, tr_sur_improve_tv, marker='o', linestyle='--', label='TR+SUR+ALB')
    plt.plot(instance, tr_mt_improve_tv, marker='^', linestyle='--', label='TR+MT+ALB')
    plt.plot(instance, tr_ms_improve_tv, marker='+', markersize='8', linestyle='--', label='TR+MS+ALB')
    line1 = plt.plot(instance[0:3], tr_st_tv[0:3], marker='o', linestyle='dotted', label='TR+ST')
    plt.plot(instance[7:], tr_st_tv[3:], marker='o', linestyle='dotted', color=line1[0].get_color())
    line2 = plt.plot(instance[0:3], tr_stmt_tv[0:3], marker='^', linestyle='dotted', label='TR+STMT')
    plt.plot(instance[7:], tr_stmt_tv[3:], marker='^', linestyle='dotted', color=line2[0].get_color())

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(instance, instance_name)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/tr_tv_all_instances.png")


def draw_tr_tv_instance():
    tr_tv = [0.523, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 2.744, 0.677]
    tr_sur_tv = [54, 32.4, 43.6, 24, 116, 276, 480, 36, 280]
    tr_mt_tv = [6, 6, 6, 6, 21, 36, 51, 8, 72]
    tr_ms_tv = [10, 10, 10, 15, 38, 40, 39, 24, 288]
    tr_sur_improve_tv = [8, 14, 40.4, 7, 24, 256, 471, 32, 378]
    tr_mt_improve_tv = [4, 6, 6, 10, 16, 34, 49, 2, 70]
    tr_ms_improve_tv = [10, 10, 10, 20, 38, 39, 40, 22, 290]
    tr_st_tv = [2, 6, 6, None, None, None, None, 8, 48]
    tr_stmt_tv = [2, 6, 6, None, None, None, None, 8, 64]

    all_methods = [tr_tv, tr_sur_tv, tr_mt_tv, tr_ms_tv, tr_sur_improve_tv, tr_mt_improve_tv, tr_ms_improve_tv,
                   tr_st_tv, tr_stmt_tv]
    plt.figure(figsize=(8, 6), dpi=300)
    methods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    method_name = ["TR", "TR+SUR", "TR+MT", "TR+MS", "TR+SUR+ALB", "TR+MT+ALB", "TR+MS+ALB", "TR+ST", "TR+STMT"]

    plt.plot(methods, [np.log10(method[0]) for method in all_methods], marker='o', linestyle='-', label="Energy2")
    plt.plot(methods, [np.log10(method[1]) for method in all_methods], marker='^', linestyle='-', label="Energy4")
    plt.plot(methods, [np.log10(method[2]) for method in all_methods], marker='+', markersize='8', linestyle='-',
             label="Energy6")
    plt.plot(methods[:-2], [np.log10(method[3]) for method in all_methods[:-2]], marker='o', linestyle='--', label="CNOT5")
    plt.plot(methods[:-2], [np.log10(method[4]) for method in all_methods[:-2]], marker='^', linestyle='--', label="CNOT10")
    plt.plot(methods[:-2], [np.log10(method[5]) for method in all_methods[:-2]], marker='+', markersize='8', linestyle='--',
             label="CNOT15")
    plt.plot(methods[:-2], [np.log10(method[6]) for method in all_methods[:-2]], marker='s', linestyle='--', label='CNOT20')
    plt.plot(methods, [np.log10(method[7]) for method in all_methods], marker='o', linestyle='dotted', label="CircuitH2")
    plt.plot(methods, [np.log10(method[8]) for method in all_methods], marker='^', linestyle='dotted', label="CircuitLiH")

    x_loc = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_loc)
    plt.xticks(methods, method_name, rotation=-15)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(fontsize=8)
    plt.tight_layout()
    # plt.show()

    plt.savefig("../figure_paper/tr_tv_per_instance_log10.png")

def draw_tr_selected():
    tr = [0.999, 0.862748429, 0.791003234, 0.876, 1.000, 1.000, 1.000, 1.000, 0.999]
    tr_sur = [0.995, 0.841779059, 0.7838377, 0.668, 0.998, 0.998, 0.999, 0.973, 0.818]
    tr_mt = [0.841, 0.682912135, 0.3832050, 0.715, 0.677, 0.710, 0.686, 0.409, 0.034]
    tr_ms = [0.96, 0.83739821, 0.7419845, 0.407, 0.981, 0.716, 0.303, 0.962, 0.776]
    tr_sur_improve = [0.995, 0.838262615, 0.7837403, 0.794, 0.998, 1.000, 0.999, 0.987, 0.917]
    tr_mt_improve = [0.997, 0.801350332, 0.5713352, 0.804, 0.991, 0.994, 0.999, 0.993, 0.504]
    tr_ms_improve = [0.998, 0.840041622, 0.7699510, 0.827, 0.999, 0.997, 0.997, 0.997, 0.88]
    tr_st = [0, 0.838716809, 0.6789286, None, None, None, None, 0.995, 0.999]
    tr_stmt = [0, 0.803798946, 0.6242092, None, None, None, None, 0.995, 0.593]

    tr_tv = [0.523, 2.752, 3.237, 6.094, 11.056, 16.795, 15.099, 2.744, 0.677]
    tr_sur_tv = [54, 32.4, 43.6, 24, 116, 276, 480, 36, 380]
    tr_mt_tv = [6, 6, 6, 6, 21, 36, 51, 8, 72]
    tr_ms_tv = [10, 10, 10, 15, 38, 40, 39, 24, 288]
    tr_sur_improve_tv = [8, 14, 40.4, 7, 24, 256, 471, 32, 378]
    tr_mt_improve_tv = [4, 6, 6, 10, 16, 34, 49, 2, 70]
    tr_ms_improve_tv = [10, 10, 10, 20, 38, 39, 40, 22, 290]
    tr_st_tv = [2, 6, 6, None, None, None, None, 8, 48]
    tr_stmt_tv = [2, 6, 6, None, None, None, None, 8, 64]

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)
    instance_name = ["Energy2", "Energy4", "Energy6", "CNOT5", "CNOT10", "CNOT15", "CNOT20", "CircuitH2", "CircuitLiH"]
    models = np.array([1, 2, 3])
    model_name = ["Continuous", "Rounding", "Improvement"]
    methods_sur = [tr, tr_sur, tr_sur_improve]
    methods_mt = [tr, tr_mt, tr_mt_improve]
    methods_ms = [tr, tr_ms, tr_ms_improve]
    methods_st = [tr, tr_st]
    methods_stmt = [tr, tr_stmt]

    methods_sur_tv = [tr_tv, tr_sur_tv, tr_sur_improve_tv]
    methods_mt_tv = [tr_tv, tr_mt_tv, tr_mt_improve_tv]
    methods_ms_tv = [tr_tv, tr_ms_tv, tr_ms_improve_tv]
    methods_st_tv = [tr_tv, tr_st_tv]
    methods_stmt_tv = [tr_tv, tr_stmt_tv]

    select = [2, 6, 8]
    # all_methods = [pgrape, pgrape_sur, pgrape_mt, pgrape_ms, pgrape_sur_improve, pgrape_mt_improve, pgrape_ms_improve,
    #                pgrape_st, pgrape_stmt]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o', label='SUR')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^', label='MT')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [1 - method[select[i]] for method in methods_sur], marker='o')
            plt.plot(models, [1 - method[select[i]] for method in methods_mt], marker='^')
            plt.plot(models, [1 - method[select[i]] for method in methods_ms], marker='+', markersize=10)

        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/tr_obj_select.png")

    fig = plt.figure(figsize=(9, 3), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.2)

    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        # ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
        if i < 9:
            ax.set_title(instance_name[select[i]])

        if i == 0:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o', label='SUR')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^', label='MT')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10, label='MS')
        else:
            plt.plot(models, [method[select[i]] for method in methods_sur_tv], marker='o')
            plt.plot(models, [method[select[i]] for method in methods_mt_tv], marker='^')
            plt.plot(models, [method[select[i]] for method in methods_ms_tv], marker='+', markersize=10)
        x_loc = plt.MultipleLocator(1)
        ax.xaxis.set_major_locator(x_loc)
        plt.xticks(models, model_name, rotation=-5)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    lines, labels = fig.axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, bbox_to_anchor=(0.35, 0, 0.3, 0.2), loc='lower center', mode='expand', borderaxespad=0,
               ncol=3, prop={'size': 10})

    plt.savefig("../figure_paper/tr_tv_select.png")


if __name__ == '__main__':
    # evo_time = 4
    # n_ts = 80
    # control = np.loadtxt("../example/control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv",
    #                      delimiter=',')
    # output_fig = "../example/figure/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_sos1.png"
    # draw_control(evo_time, n_ts, control, output_fig)
    # draw_stats()
    # draw_sos1(n_ts, control, output_fig)
    # print(max([abs(sum(control[k, :]) - 1) for k in range(n_ts)]))
    # draw_integral_error("H2", ub=True)
    draw_stats()
    # draw_obj_energy_r()
    # draw_sur()
    # draw_mt()
    # draw_ms()
    # draw_sur_improve()
    # draw_mt_improve()
    # draw_ms_improve()
    # draw_grape_obj()
    # draw_pgrape_obj()
    # draw_admm_obj()
    # draw_tr_obj()
    # draw_grape_tv()
    # draw_pgrape_tv()
    # draw_admm_tv()
    # draw_tr_tv()
    # draw_grape_obj_instance()
    # draw_grape_tv_instance()
    # draw_pgrape_obj_instance()
    # draw_pgrape_tv_instance()
    # draw_pgrape_obj_tv_instance_split()
    # draw_pgrape_selected()
    # draw_admm_obj_instance()
    # draw_admm_tv_instance()
    # draw_admm_instance_split()
    # draw_admm_obj_tv_instance_split()
    # draw_admm_selected()
    # draw_tr_obj_instance()
    # draw_tr_tv_instance()
    # draw_tr_obj_tv_instance_split()
    # draw_tr_selected()
