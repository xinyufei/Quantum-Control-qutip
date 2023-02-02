#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sum_penalty=1e-4 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --sum_penalty=1e-4
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.0001.csv" \
#    --sum_penalty=1e-3
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.001.csv" \
#    --sum_penalty=1e-2
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --sum_penalty=1e-1
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --sum_penalty=1
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sum_penalty=10
cd ../ADMM/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sum_penalty=1.0 --alpha=1e-8 --rho=1 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sum_penalty=1.0 --alpha=1e-7 --rho=1 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sum_penalty=1.0 --alpha=1e-6 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-5 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-4 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-2 --rho=0.5 --max_iter_admm=100
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.01_ADMM_0.5_iter100.csv" \
#    --sum_penalty=1.0 --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Continuous/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --sos=1 --alpha=0.0001 --tr_type="tvc"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --sos=1 --alpha=0.0001 --tr_type="tvc"

cd ../SwitchingTime/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --min_up_time=0 --alpha=0.0001
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --min_up_time=0.5 --alpha=0.0001
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/Trustregion/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0 --alpha=0.0001
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 --admm_control="../control/Trustregion/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --min_up_time=0.5 --alpha=0.0001
    
cd ../Rounding
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Continuous/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/ADMM/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Trustregion/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --type=SUR

#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Continuous/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/ADMM/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Trustregion/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5

#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Continuous/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=20
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/ADMM/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=20
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_control="../control/Trustregion/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=20

cd ../Trustregion/
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_1_SUR.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_1_SUR.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_1_SUR.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_minup5_0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_minup5_0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
#    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5_0.csv" \
#    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5

python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_maxswitch20_0.csv" \
    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
    --initial_file="../control/Rounding/MoleculeADMMVQENew_H2_evotime10.0_n_ts100_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100_maxswitch20_0.csv" \
    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=10 --n_ts=100 \
    --initial_file="../control/Rounding/MoleculeVQENew_H2_evotime10.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty1.0_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch20_0.csv" \
    --target="../../hamiltonians/H2_target.csv" --sos1=1 --alpha=0.001 --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=20
#
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup10.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=10
#
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch8.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_maxswitch8.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100_maxswitch8.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8
#python Molecule.py --molecule=H2 --qubit_num=2 --evo_time=4 --n_ts=80 \
#    --initial_file="../control/Rounding/Molecule_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch8.csv" \
#    --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=8


cd ../../scripts/linux