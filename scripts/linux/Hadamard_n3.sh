#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sum_penalty=1e-4 --max_iter=3000
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.0001.csv" \
#    --sum_penalty=1e-3 --max_iter=3000
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.001.csv" \
#    --sum_penalty=1e-2 --max_iter=3000
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sum_penalty=1e-1 --max_iter=3000
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --sum_penalty=1 --max_iter=3000
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sum_penalty=10 --max_iter=3000
cd ../ADMM/
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sum_penalty=0.01 --alpha=1e-6 --rho=1 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/ADMM/HadamardADMM2_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty1e-06_ADMM_1.0_iter100.csv" \
#    --sum_penalty=0.01 --alpha=1e-5 --rho=0.5 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sum_penalty=0.01 --alpha=1e-5 --rho=0.5 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --sum_penalty=0.01 --alpha=1e-4 --rho=0.5 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sum_penalty=0.01 --alpha=1e-3 --rho=0.5 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.001_ADMM_0.5_iter100.csv" \
#    --sum_penalty=0.01 --alpha=1e-2 --rho=0.5 --max_iter_admm=100
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --initial_type=WARM \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_0.5_iter100.csv" \
#    --sum_penalty=0.01 --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sos=1 --alpha=0.0001 --tr_type="tvc"
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/ADMM/HadamardADMM2_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos=1 --alpha=0.0001 --tr_type="tvc"

cd ../SwitchingTime/
python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --admm_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
    --min_up_time=0 --alpha=0.0001
python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --admm_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
    --min_up_time=1 --alpha=0.0001
python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --admm_control="../control/Trustregion/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
    --min_up_time=0 --alpha=0.0001
python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 --admm_control="../control/Trustregion/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
    --min_up_time=1 --alpha=0.0001
    
cd ../Rounding
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sos1=0 --t_sos=1 --type=SUR
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=12

#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sos1=0 --t_sos=1 --type=SUR
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Continuous/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=12
####
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type=SUR
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM2_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.001_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/ADMM/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=12
####
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Trustregion/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --t_sos=1 --type=SUR
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Trustregion/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_control="../control/Trustregion/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=12

cd ../Trustregion/
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_SUR.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_SUR.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100_SUR.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="tv"
#
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch12.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_maxswitch12.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
    --initial_file="../control/Rounding/HadamardADMM3_evotime12.0_n_ts120_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.0001_ADMM_0.5_iter100_maxswitch12.csv" \
    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
#python Hadamard.py --qubit_num=3 --evo_time=12 --n_ts=120 \
#    --initial_file="../control/Rounding/Hadamard3_evotime12.0_n_ts120_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch12.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12


cd ../../scripts/linux