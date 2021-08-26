#!/user/bin/env bash
cd "/Volumes/GoogleDrive/My Drive/PhD-topic/Quantum-Computing/code/Quantum-Control-qutip/example/Trustregion/"
conda activate qcopt
python energy.py --min_up=10
cd ../../scripts