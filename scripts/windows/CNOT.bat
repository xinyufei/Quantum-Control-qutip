conda activate pyqcopt
cd ../../example/Rounding
python CNOT.py --evo_time=10 --n_ts=200^
    --initial_control="../control/Continuous/CNOT_evotime10.0_n_ts200_ptypeCONSTANT_offset0.5_objUNIT.csv"^
    --sos1=False --type=SUR
cd ../../scripts/windows