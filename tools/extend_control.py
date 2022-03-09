import numpy as np


def extend_control(initial_control_name, pre_num_step, cur_num_step):
    new_control_name = initial_control_name.split(".csv")[0] + "_extend_ts_" + str(cur_num_step) + ".csv"
    initial_control = np.loadtxt(initial_control_name, delimiter=",")
    step = int(cur_num_step / pre_num_step)

    cur_control = np.zeros((cur_num_step, initial_control.shape[1]))

    for i in range(pre_num_step):
        for j in range(step):
            cur_control[i*step+j, :] = initial_control[i, :]

    np.savetxt(new_control_name, cur_control, delimiter=",")


if __name__ == '__main__':
    # extend_control("../example/control/ADMM/MoleculeADMM_BeH2_evotime5.0_n_ts50_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.001_ADMM_3.0_iter100.csv",
    #                50, 300)
    # extend_control("../example/control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv",
    #                40, 80)
    extend_control("../example/control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv", 
                   80, 240)