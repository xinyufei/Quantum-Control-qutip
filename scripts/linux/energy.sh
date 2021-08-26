#!/user/bin/env bash

cd ../../example/Continuous/
conda activate qcopt
#python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
cd ../ADMM/
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
    --initial_control="../control/Continuous/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5.csv" \
    --alpha=1e-4 --rho=10 --max_iter_admm=100
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.0001_ADMM_10.0_iter100.csv" \
    --alpha=1e-3 --rho=10 --max_iter_admm=100
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.001_ADMM_10.0_iter100.csv" \
    --alpha=1e-2 --rho=10 --max_iter_admm=100
python energy.py --n=2 --num_edges=1 --evo_time=2 --n_ts=40 --initial_type=WARM \
    --initial_control="../control/ADMM/EnergyADMM2_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100.csv" \
    --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../../scripts/linux