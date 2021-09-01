#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=CONSTANT --offset=0.5
cd ../ADMM/
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --alpha=1e-4 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.0001_ADMM_10.0_iter100.csv" \
#    --alpha=1e-3 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.001_ADMM_10.0_iter100.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --alpha=1e-1 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --alpha=1 --rho=10 --max_iter_admm=100
cd ../Rounding/
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --type=maxswitch --max_switch=5
##
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_alpha0.1_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_alpha0.1_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Trustregion/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_alpha0.1_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=maxswitch --max_switch=5

cd ../SwitchingTime/
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --admm_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --min_up_time=0 --alpha=0.1
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --admm_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --min_up_time=0.5 --alpha=0.1
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --admm_control="../control/Trustregion/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_alpha0.1_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0 --alpha=0.1
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --admm_control="../control/Trustregion/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_alpha0.1_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0.5 --alpha=0.1
cd ../Trustregion/
#conda activate qcopt
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100_SUR.csv" \
#    --alpha=0.1 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_SUR.csv" \
#    --alpha=0.1 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100_minup10.csv" \
#    --alpha=0.1 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_minup10.csv" \
#    --alpha=0.1 --tr_type="hard" --hard_type="minup" --min_up=10
python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
    --initial_file="../control/Rounding/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100_maxswitch5.csv" \
    --alpha=0.1 --tr_type="hard" --hard_type="maxswitch" --max_switch=5
python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
    --initial_file="../control/Rounding/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5_maxswitch5.csv" \
    --alpha=0.1 --tr_type="hard" --hard_type="maxswitch" --max_switch=5

#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/SwitchTime/EnergyST2_evotime_1.0_n_ts20_n_switch1_initwarm_minuptime0.0.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/SwitchTime/EnergyST2_evotime_1.0_n_ts20_n_switch1_initwarm_minuptime0.5.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
    --initial_file="../control/SwitchTime/EnergyST2_evotime_1.0_n_ts20_n_switch1_initwarm_minuptime0.0.csv" \
    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5

#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --alpha=0.01 --tr_type="tvc"
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --alpha=1 --tr_type="tvc"

cd ../../scripts/linux