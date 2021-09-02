#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=CONSTANT --offset=0.5
#cd ../ADMM/
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/Continuous/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --alpha=1e-6 --rho=0.25 --max_iter_admm=100
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty1e-06_ADMM_0.25_iter100.csv" \
#    --alpha=1e-5 --rho=0.25 --max_iter_admm=100
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty1e-05_ADMM_0.25_iter100.csv" \
#    --alpha=1e-4 --rho=0.25 --max_iter_admm=100
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.0001_ADMM_0.25_iter100.csv" \
#    --alpha=1e-3 --rho=0.25 --max_iter_admm=100
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --alpha=1e-2 --rho=0.25 --max_iter_admm=100
#python CNOT.py --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
#    --alpha=1e-1 --rho=0.25 --max_iter_admm=100
cd ../Trustregion/
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Continuous/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos=0 --alpha=0.01 --tr_type="tvc"
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
#    --sos=0 --alpha=0.01 --tr_type="tvc"

cd ../Rounding
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type=SUR
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/Continuous/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT.csv" \
    --sos1=0 --type="minup" --min_up=10
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20
##
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type=SUR
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty1e-05_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=10
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.0001_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=10
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=10
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=10
python CNOT.py --evo_time=1 --n_ts=20 \
    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.1_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=10
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20
##
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type=SUR
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="minup" --min_up=10
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20

cd ../Trustregion/
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOT_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_objUNIT_maxswitch20.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#python CNOT.py --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/CNOTADMM_evotime1.0_n_ts20_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100_maxswitch20.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=20


cd ../../scripts/linux