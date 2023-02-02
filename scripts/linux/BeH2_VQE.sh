#!/user/bin/env bash

conda activate qcopt
cd ../../example/Continuous/
#python Molecule.py --gen_target=1 --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=5 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0.01 \
#    --target="../../hamiltonians/BeH2_optimize_target.csv" --max_iter=10000
#python Molecule.py --gen_target=0 --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --max_iter=10000
#python Molecule.py --gen_target=0 --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=800 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --max_iter=10000
#python Molecule.py --gen_target=0 --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=600 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --max_iter=10000
#python Molecule.py --gen_target=0 --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=1000 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0 \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --max_iter=10000
#python Molecule.py --name=MoleculeVQE2 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0.1 --max_iter=10000
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime20.0_n_ts200_target.csv"
#python Molecule.py --gen_target=0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=2000 --initial_type=CONSTANT --offset=0.5 --sum_penalty=0.1 --max_iter=10000\
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv"
#python Molecule.py --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sum_penalty=0.01 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=5 --n_ts=20 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts100_target.csv" --sum_penalty=0.01 --max_iter=10000
#python Molecule.py --gen_target=1 --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=5 --n_ts=50 --initial_type=CONSTANT --sum_penalty=0.01 --max_iter=10000
#python Molecule.py --gen_target=0 --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=3 --n_ts=60 --initial_type=CONSTANT --sum_penalty=1.0 --max_iter=10000 \
#  --target="../control/Continuous/MoleculeVQE_BeH2_evotime3.0_n_ts60_target.csv"
#python Molecule.py --name=MoleculeVQE --molecule=BeH2 --qubit_num=6 --evo_time=3 --n_ts=60 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime3.0_n_ts60_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime3.0_n_ts60_target.csv" --sum_penalty=1.0 --max_iter=10000
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime1.0_n_ts20_target.csv"
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1e-6 --max_iter=10000
#python Molecule.py --name=MoleculeVQE2 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=100 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts100_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=5000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=400 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts400_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=800 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts800_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=600 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts600_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=1000 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts1000_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.001.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-2
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-1
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=10
#
#python Molecule.py --name=MoleculeVQE2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-4
#python Molecule.py --name=MoleculeVQE2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1e-3
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1e-2 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1e-1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1 --max_iter=10000
#python Molecule.py --name=MoleculeVQE2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=10

cd ../ADMM/
python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 --initial_type=WARM \
    --initial_control="../control/Continuous/MoleculeVQENew_BeH2_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.01.csv" \
    --target="../../hamiltonians/BeH2_target.csv" --sum_penalty=0.01 --alpha=0.01 --rho=3.0 --max_iter_admm=50
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=40 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts40_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sum_penalty=0.01 --alpha=0.01 --rho=3.0 --max_iter_admm=50
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=5 --n_ts=100 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts100_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts50_target.csv" --sum_penalty=0.01 --alpha=1e-3 --rho=3.0 --max_iter_admm=50
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=5 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime5.0_n_ts50_target.csv" --sum_penalty=0.01 --alpha=1e-3 --rho=3.0 --max_iter_admm=30
#python Molecule.py --name=MoleculeADMM2 --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-5 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeADMMNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_objUNIT_sum_penalty1.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-5 --rho=0.5 --max_iter_admm=100
#python Molecule.py --name=MoleculeADMMNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-5 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=3.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-2 --rho=1.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeADMMNew3 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE2_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeADMMNew3 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-2 --rho=5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeADMMNew --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --name=MoleculeVQEADMM0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=1.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=5.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1.0 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM0 --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=1.0 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.01 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.01 --alpha=1e-5 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.01 --alpha=1e-2 --rho=5.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMMW --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-4 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMMW --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeVQEADMMW_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-3 --rho=0.5 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMMW --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeVQEADMMW_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-2 --rho=5.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeVQEADMMW --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeVQEADMMW_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.01_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sum_penalty=0.1 --alpha=1e-1 --rho=10.0 --max_iter_admm=100 --max_iter_step=5000
#python Molecule.py --name=MoleculeADMMNew2W --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMMNew2W_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=0.1 --alpha=1e-2 --rho=0.5 --max_iter_admm=100 --max_iter_step=3000
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 --initial_type=WARM \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.01_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sum_penalty=1.0 --alpha=1e-1 --rho=10 --max_iter_admm=100
cd ../Trustregion/
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos=1 --alpha=0.01 --tr_type="tvc" --sum_penalty=0.01
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos=1 --alpha=0.0001 --tr_type="tvc" --sum_penalty=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos=1 --alpha=0.001 --tr_type="tvc" --sum_penalty=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_file="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos=1 --alpha=0.0001 --tr_type="tvc"

cd ../SwitchingTime/
#python Molecule.py --name=MoleculeSTVQEADMM --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --admm_control="../control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --min_up_time=0 --alpha=0.001
#python Molecule.py --name=MoleculeSTVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --admm_control="../control/ADMM/MoleculeVQEADMM0_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --min_up_time=0.5 --alpha=0.001
#python Molecule.py --name=MoleculeSTVQETR --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --admm_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --min_up_time=0 --alpha=0.001
#python Molecule.py --name=MoleculeSTVQE --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --admm_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --min_up_time=0.5 --alpha=0.001
    
cd ../Rounding
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40

#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=100 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts100_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=400 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts400_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=100 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts100_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=400 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts400_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=600 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts600_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=800 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts800_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=1000 --time_limit=300 \
#    --initial_control="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts1000_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --t_sos=1 --type=SUR
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --type="SUR"
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40
####
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40
#python Molecule.py --molecule=LiH --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.001_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty1e-06_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_LiH_evotime15.0_n_ts150_ptypeWARM_offset0.5_sum_penalty0.1_penalty1e-05_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMMNew3_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.001_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeADMMNew2W_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.01_ADMM_0.5_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE2_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=15 --n_ts=150 \
#    --initial_control="../control/ADMM/MoleculeADMM_H2_evotime4.0_n_ts80_ptypeWARM_offset0.5_sum_penalty1.0_penalty0.1_ADMM_10.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime15.0_n_ts150_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=10
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/ADMM/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40
##
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40
#
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --type=SUR
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="minup" --min_up=5
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_control="../control/Trustregion/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=0 --t_sos=1 --type="maxswitch" --max_switch=40

cd ../Trustregion/
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeADMM_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv" --eta=0.0001
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM0_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="tv"
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.0001_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="tv"

#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_minup5_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100_minup5_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=5
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=5

#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch40.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_maxswitch40_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeADMM_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.01_ADMM_3.0_iter100_maxswitch40_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=40
#python Molecule.py --molecule=BeH2 --qubit_num=6 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_BeH2_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.01_alpha0.01_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch40_0.csv" \
#    --target="../control/Continuous/MoleculeVQE_BeH2_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.01 --tr_type="hard" --hard_type="maxswitch" --max_switch=40


#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv" --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv" --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv" --eta=0.0001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_SUR.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="tv" --eta=0.0001

#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.001_ADMM_3.0_iter100_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.25
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM0_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.0001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM0_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.00001
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_minup5.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="minup" --min_up=5 --eta=0.25

#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT_sum_penalty0.0_maxswitch40.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40 --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_maxswitch40.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.0001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40 --eta=0.1
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQEADMM0_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_sum_penalty0.1_penalty0.0001_ADMM_0.5_iter100_maxswitch40.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40 --eta=0.25
#python Molecule.py --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=200 \
#    --initial_file="../control/Rounding/MoleculeVQE_LiH_evotime20.0_n_ts200_ptypeWARM_offset0.5_objUNIT_sum_penalty0.1_alpha0.001_sigma0.25_eta0.001_threshold30_iter100_typetvc_maxswitch40.csv" \
#    --target="../control/Continuous/MoleculeVQE_LiH_evotime20.0_n_ts200_target.csv" --sos1=1 --alpha=0.001 --tr_type="hard" --hard_type="maxswitch" --max_switch=40 --eta=0.25


cd ../../scripts/linux