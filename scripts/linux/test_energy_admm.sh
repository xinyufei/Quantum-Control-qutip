#!/user/bin/env bash
cd "/Volumes/GoogleDrive/My Drive/PhD-topic/Quantum-Computing/code/Quantum-Control-qutip/example/ADMM"
conda activate qcopt
python energy.py --alpha=0.001 --rho=0.25 --max_iter_admm=100 >> "../output/ADMM/Energy_2_2_40_0.001_0.05.log"
python energy.py --alpha=0.001 --rho=0.0001 >> "../output/ADMM/Energy_2_2_40_0.001_0.0001.log"
python energy.py --alpha=0.001 --rho=0.01 >> "../output/ADMM/Energy_2_2_40_0.001_0.01.log"
python energy.py --alpha=0.001 --rho=0.1 >> "../output/ADMM/Energy_2_2_40_0.001_0.1.log"
python energy.py --alpha=0.001 --rho=0.5 >> "../output/ADMM/Energy_2_2_40_0.001_0.5.log"
cd ../../scripts