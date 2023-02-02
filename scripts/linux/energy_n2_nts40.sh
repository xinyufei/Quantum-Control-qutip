#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
python energy.py --name=EnergyAcc --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=32 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=64 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=128 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=256 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=512 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=1024 --initial_type=CONSTANT --offset=0.5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=2048 --initial_type=CONSTANT --offset=0.5
cd ../ADMM/
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=32 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=5 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=64 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=5 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=128 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=50 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=256 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=50 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=512 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=50 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=1024 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=40 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=2048 --initial_type=WARM \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts32_ptypeCONSTANT_offset0.5_instance0.csv" \
#    --alpha=1e-2 --rho=60 --max_iter_admm=50
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.0001_ADMM_10.0_iter100.csv" \
#    --alpha=1e-3 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.001_ADMM_10.0_iter100.csv" \
#    --alpha=1e-2 --rho=10 --max_iter_admm=100
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Rounding/
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/Continuous/Energy2_evotime1.0_n_ts20_ptypeCONSTANT_offset0.5.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=SUR
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=minup --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --type=maxswitch --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_muscod.csv" \
#    --type=SUR

cd ../SwitchingTime/
#python energy.py --name='EnergySTADMM' --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --admm_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --min_up_time=0 --alpha=0.01
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --admm_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
#    --min_up_time=0.5 --alpha=0.01
#python energy.py --name='EnergySTTR' --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --admm_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0 --alpha=0.01
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --admm_control="../control/Trustregion/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0.5 --alpha=0.01
cd ../Trustregion/
#conda activate qcopt
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100_SUR.csv" \
#    --alpha=0.1 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_maxswitch5.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_maxswitch5.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5

#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/SwitchTime/EnergyST2_evotime_1.0_n_ts40_n_switch0_initwarm_minuptime0.0.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/SwitchTime/EnergyST2_evotime_1.0_n_ts40_n_switch0_initwarm_minuptime0.5.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/SwitchTime/EnergyST2_evotime_2.0_n_ts40_n_switch0_initwarm_minuptime0.0.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --alpha=0.01 --tr_type="tv"
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 \
#    --initial_file="../control/Rounding/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch5.csv" \
#    --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=5

#python energy.py --n=2 --num_edges=1 --evo_time=1 --n_ts=20 \
#    --initial_file="../control/ADMM/EnergyADMM2_evotime1.0_n_ts20_ptypeWARM_offset0.5_penalty0.1_ADMM_10.0_iter100.csv" \
#    --alpha=0.01 --tr_type="tvc"


cd ../../scripts/linux