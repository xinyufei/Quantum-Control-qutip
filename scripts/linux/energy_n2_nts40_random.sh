#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=1 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=2 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=3 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
cd ../ADMM/
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance2.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance3.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --alpha=1e-1 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty1e-05_ADMM_10.0_iter100.csv" \
#    --alpha=1e-4 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty1e-05_ADMM_10.0_iter100.csv" \
#    --alpha=1e-3 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.001_ADMM_10.0_iter100.csv" \
#    --alpha=1e-2 --rho=15 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
#    --alpha=0.01 --tr_type="tvc"
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=2 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance2.csv" \
#    --alpha=0.01 --tr_type="tvc"
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=3 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance3.csv" \
#    --alpha=0.01 --tr_type="tvc"
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4.csv" \
#    --alpha=0.01 --tr_type="tvc"
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --alpha=0.01 --tr_type="tvc"
cd ../Rounding/
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance2.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance3.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance2.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance3.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance2.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance3.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance2.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance3.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance2.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance3.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=2 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance2.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=3 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance3.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=4 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance1.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=3 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=3 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=3 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --type=SUR
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --rgraph=1 --seed=5 --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100_instance5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.001_ADMM_10.0_iter100.csv" \
#    --type=minup --min_up=5
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --type=minup --min_up=5
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM6_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=SUR
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=SUR
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2   --num_edges=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2   --num_edges=2 --seed=2 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy6_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_muscod_long.csv" \
#    --type=SUR

cd ../SwitchingTime/
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 \
#    --admm_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --min_up_time=0 --alpha=0.01
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 \
#    --admm_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance5.csv" \
#    --min_up_time=0.5 --alpha=0.01
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 \
#    --admm_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0 --alpha=0.01
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=5 --evo_time=2 --n_ts=40 \
#    --admm_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance5_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0.5 --alpha=0.01
cd ../Trustregion/
#conda activate qcopt
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4_1_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4_1_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_1_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#
#python energy.py --n=2 --num_edges=1 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4_minup10_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4_minup10_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4_maxswitch5_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4_maxswitch5_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5
#python energy.py --n=2   --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance4_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch5_1.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5


cd ../../scripts/linux