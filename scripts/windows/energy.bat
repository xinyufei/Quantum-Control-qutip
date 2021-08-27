conda activate pyqcopt
cd ../../example/Rounding
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv"^
    --type=SUR
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv"^
    --type=minup --min_up=10
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv"^
    --type=maxswitch --max_switch=5
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv"^
    --type=SUR
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv"^
    --type=minup --min_up=10
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40^
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv"^
    --type=maxswitch --max_switch=5
cd ../../scripts/windows