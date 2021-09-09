#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv"
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-4
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.0001.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-3
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.001.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-2
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-1
#python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1
python Molecule.py --name=MoleculeNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=10
#
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-4
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-3
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-2
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-1
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1
python Molecule.py --name=MoleculeNew2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=10

cd ../ADMM/
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-6 --rho=0.5 --max_iter_admm=100
#python Molecule.py --name=MoleculeADMM2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-5 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeADMM2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeADMM2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeADMM2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-2 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.01_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos=1 --alpha=0.0001 --tr_type="tvc" --sum_penalty=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos=1 --alpha=0.0001 --tr_type="tvc"

cd ../SwitchingTime/
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --admm_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --min_up_time=0 --alpha=0.0001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --admm_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --min_up_time=0.5 --alpha=0.0001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --admm_control="../control/Trustregion/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --min_up_time=0 --alpha=0.0001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --admm_control="../control/Trustregion/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --min_up_time=0.5 --alpha=0.0001
    
cd ../Rounding
#python Molecule.py --molecule=LiH -qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_control="../control/Continuous/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=30
#
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Continuous/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=30
####
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.01_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.1_ADMM_10.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=30
###
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Trustregion/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Trustregion/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/Trustregion/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=30

cd ../Trustregion/
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup5.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_minup5.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_minup5.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5
#
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch30.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=30
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_maxswitch30.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=30
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_maxswitch30.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=30
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/Rounding/Molecule3_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch30.csv" \
#    --target="../control/Continuous/MoleculeNew_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=30


cd ../../scripts/linux