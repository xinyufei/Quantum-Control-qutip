#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv"
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=40 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=5000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=160 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=5000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=240 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=5000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=320 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=5000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=960 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=10000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=32 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=10000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=400000 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=100000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=400000 --initial_type=WARM --sum_penalty=1.0 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts400000_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --max_iter=100000
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=32 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts32_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=160 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts160_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=240 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts240_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=320 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts320_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=960 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts480_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1e-6
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.0001.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1e-3
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.001.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1e-2
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1e-1
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=10
#python Molecule.py --name=MoleculeNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty10.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1e25
cd ../ADMM/
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=32 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts32_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=50
python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=1024 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts1024_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=50
python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=2048 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts2048_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=50
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=960 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts960_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=50
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-4 --rho=0.5 --max_iter_admm=100
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=100
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sum_penalty=1.0 --alpha=1e-2 --rho=0.5 --max_iter_admm=100
#python Molecule.py --name=MoleculeADMMNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.01_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos=1 --alpha=0.001 --tr_type="tvc"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos=1 --alpha=0.001 --tr_type="tvc"

cd ../SwitchingTime/
#python Molecule.py --name=MoleculeSTNewADMM --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=320 --admm_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --min_up_time=0 --alpha=0.001 --initial_type='ave'
#python Molecule.py --name=MoleculeSTNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --min_up_time=0.5 --alpha=0.001
#python Molecule.py --name=MoleculeSTNewTR --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --min_up_time=0 --alpha=0.001
#python Molecule.py --name=MoleculeSTNew --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --min_up_time=0.5 --alpha=0.001
    
cd ../Rounding
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=8
#
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=4096 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts4096_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=40 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts40_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=160 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts160_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=240 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts240_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=320 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts320_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=400 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts400_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=400000 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts400000_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=8
####
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/ADMM/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=8
###
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Trustregion/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=8

cd ../Trustregion/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
##
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup10.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_minup10.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_minup10.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=10
##
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch8.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_maxswitch8.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMMNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_maxswitch8.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeNew_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch8.csv" \
#    --target="../control/Continuous/MoleculeNew_H2_evotime4.0_n_ts80_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8


cd ../../scripts/linux