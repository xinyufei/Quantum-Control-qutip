import numpy as np
import matplotlib.pyplot as plt


def draw_stats():
    x = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    sum_norm = [167.48180578702548, 167.47867920029202, 95.40499099227462, 34.144492624667244,
                30.14356912948749, 18.70981873414983, 0.3962903548156124, 5.55381795807922e-07,
                8.428231286969557e-10]

    print([sum_norm[i] * x[i] * x[i] for i in range(len(x))])

    exit()
    # plt.plot(np.array(x), np.array(sum_norm), '-o', label='squared_L2_norm')
    # plt.plot(np.array(x), 1e-9 / np.power(np.array(x), 2))
    # plt.show()
    # exit()

    plt.figure(dpi=300)
    plt.plot(np.array(x), np.array(sum_norm), '-o', label='squared_L2_norm')
    plt.xlabel("Penalty parameter")
    plt.ylabel("Squared Penalized Term")
    plt.legend()
    plt.savefig("../example/figure/Continuous/MoleculeNew_H2_evotime4.0_n_ts80.png")

    plt.figure(dpi=300)
    plt.plot(np.log10(np.array(x)), np.array(sum_norm), '-o', label='squared_L2_norm')
    plt.xlabel("Common logarithm of penalty parameter")
    plt.ylabel("Squared Penalized Term")
    plt.legend()
    plt.savefig("../example/figure/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_log10.png")


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


if __name__ == '__main__':
    evo_time = 4
    n_ts = 80
    control = np.loadtxt("../example/control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv",
                         delimiter=',')
    output_fig = "../example/figure/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_sos1.png"
    # draw_control(evo_time, n_ts, control, output_fig)
    # draw_stats()
    draw_sos1(n_ts, control, output_fig)
    print(max([abs(sum(control[k, :]) - 1) for k in range(n_ts)]))


