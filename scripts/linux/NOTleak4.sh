#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python NOTleak.py --name=NOTleak --evo_time=4 --n_ts=40 --initial_type=CONSTANT --offset=0.5
cd ../ADMM/
#python NOTleak.py --name=NOTleakADMM --evo_time=4 --n_ts=40 --initial_type=WARM --offset=0.5 \
#    --initial_control="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --alpha=1e-5 --rho=0.25 --max_iter_admm=100
#python NOTleak.py --name=NOTleakADMM --evo_time=4 --n_ts=40 --initial_type=WARM --offset=0.5 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty1e-05_ADMM_0.25_iter100.csv" \
#    --alpha=1e-4 --rho=0.25 --max_iter_admm=100
#python NOTleak.py --name=NOTleakADMM --evo_time=4 --n_ts=40 --initial_type=WARM --offset=0.5 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.0001_ADMM_0.25_iter100.csv" \
#    --alpha=1e-3 --rho=0.25 --max_iter_admm=100
#python NOTleak.py --name=NOTleakADMM --evo_time=4 --n_ts=40 --initial_type=WARM --offset=0.5 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --alpha=1e-2 --rho=100 --max_iter_admm=100
#python NOTleak.py --name=NOTleakADMM --evo_time=4 --n_ts=40 --initial_type=WARM --offset=0.5 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_25.0_iter100.csv" \
#    --alpha=1e-1 --rho=10 --max_iter_admm=100

cd ../Trustregion/
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos=0 --alpha=0.001 --tr_type="tvc"
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Continuous/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos=0 --alpha=0.001 --tr_type="tvc"
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Continuous/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos=0 --alpha=0.001 --tr_type="tvc"
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Continuous/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos=0 --alpha=0.01 --tr_type="tvc"
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/ADMM/CNOTSUADMM_evotime5.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100.csv" \
#    --sos=0 --alpha=0.01 --tr_type="tvc"

cd ../Rounding
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="SUR"
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Trustregion/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="SUR"
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="SUR"

python NOTleak.py --evo_time=4 --n_ts=40 \
    --initial_control="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
    --sos1=0 --type="minup" --min_up=5
python NOTleak.py --evo_time=4 --n_ts=40 \
    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
    --sos1=0 --type="minup" --min_up=5
python NOTleak.py --evo_time=4 --n_ts=40 \
    --initial_control="../control/Trustregion/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
    --sos1=0 --type="minup" --min_up=5


#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/ADMM/NOTleaknewADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=10
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/Continuous/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Continuous/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=10
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Trustregion/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/Trustregion/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Trustregion/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=10

#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/ADMM/NOTleakADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Continuous/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/Continuous/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Continuous/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Trustregion/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_control="../control/Trustregion/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Trustregion/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --sos1=0 --type="maxswitch" --max_switch=20

cd ../Trustregion/
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknewADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_0_SUR.csv" \
#    --alpha=0.001 --tr_type="tv"

#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknewADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=2

#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_minup5_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5

#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_maxswitch2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_maxswitch6_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_maxswitch10_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=10
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch6_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknew_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch10_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=10
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch2_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=2
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch6_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=6
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleaknewADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch10_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=10

#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_maxswitch4_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_maxswitch12_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_maxswitch20_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleak_evotime4.0_n_ts40_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch4_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleak_evotime6.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch12_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleak_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch20_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#python NOTleak.py --evo_time=4 --n_ts=40 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch4_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=4
#python NOTleak.py --evo_time=6 --n_ts=60 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime6.0_n_ts60_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch12_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=12
#python NOTleak.py --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/NOTleakADMM_evotime10.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.001_ADMM_0.25_iter100_maxswitch20_0.csv" \
#    --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20

#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSUADMM_evotime5.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_maxswitch20.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSUADMM_evotime5.0_n_ts100_ptypeWARM_offset0.5_objUNIT_penalty0.01_ADMM_0.25_iter100_maxswitch20.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python CNOT.py --evo_time=5 --n_ts=100 \
#    --initial_file="../control/Rounding/CNOTSU_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch20.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=20


cd ../../scripts/linux