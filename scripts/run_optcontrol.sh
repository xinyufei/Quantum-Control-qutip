#!/user/bin/env bash
cd "/Volumes/GoogleDrive/My Drive/PhD-topic/Quantum-Computing/code/Quantum-Control-qutip/example"
conda activate qcopt
# python CNOT.py --name="CNOT" --evo_time=1 --n_ts=20 --initial_type='CONSTANT' --offset=0.5 >> "../output/CNOT_1_20_CONSTANT_0.5.log"
# python CNOT.py --name="CNOT" --evo_time=5 --n_ts=100 --initial_type='CONSTANT' --offset=0.5 >> "../output/CNOT_5_100_CONSTANT_0.5.log"
# python CNOT.py --name="CNOT" --evo_time=10 --n_ts=200 --initial_type='CONSTANT' --offset=0.5 >> "../output/CNOT_10_200_CONSTANT_0.5.log"
# python CNOT.py --name="CNOT" --evo_time=20 --n_ts=400 --initial_type='CONSTANT' --offset=0.5 >> "../output/CNOT_20_400_CONSTANT_0.5.log"
python spin_chain_Hadamard_RWA.py --qubit_num=2 --evo_time=8 --n_ts=80 --initial_type='CONSTANT' --offset=0.5 >> "../output/Hadamard2_8_80_CONSTANT_0.5.log"
cd ../scripts