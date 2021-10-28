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
    #             8.822329350354554e-08, 1.4531085324239638e-08]

    print([sum_norm[i] * x[i] * x[i] for i in range(len(x))])

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
             label=r'$\log_{10}$' + str(round(sum_norm[constant_idx], 2)) + r'$-\log_{10} \rho$')
    plt.xlabel("Common logarithm of penalty parameter")
    plt.ylabel("Common logarithm of squared penalized Term")
    plt.legend()
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
    draw_integral_error("H2", ub=True)
    # draw_stats()
    # draw_obj_energy_r()
