#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python CNOT.py --evo_time=10 --n_ts=200 --initial_type=CONSTANT --offset=0.5
#cd ../ADMM/
#python CNOT.py --evo_time=10 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/CNOT_evotime10.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT.csv" \
#    --alpha=1e-4 --rho=0.25 --max_iter_admm=200