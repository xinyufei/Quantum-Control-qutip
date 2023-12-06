# Binary Control Pulse Optimization for Quantum Systems
This repository contains the source code used in the computational experiments of the paper: 
**Binary Control Pulse Optimization for Quantum Systems**. 

In our paper, we first apply GRAPE algorithm to solve the continuous relaxation of binary quantum control problems 
with SOS1 property. Then we apply two rounding techniques, sum-up rounding (SUR) and combinatorial integral 
approximation (CIA) to obtain the binary controls. 
Furthermore, we apply our improvement heuristic to improve the solutions. 

## Citation
If you use our code in your research, please cite our paper:
> [**Binary Control Pulse Optimization for Quantum Systems**](https://quantum-journal.org/papers/q-2023-01-04-892/pdf/) <br />
> Xinyu Fei, Lucas T. Brady, Jeffrey Larson, Sven Leyffer, Siqian Shen <br />
> *Quantum, 2023*
> ```
> @article{fei2023binary,
>   title={Binary control pulse optimization for quantum systems},
>   author={Fei, Xinyu and Brady, Lucas T and Larson, Jeffrey and Leyffer, Sven and Shen, Siqian},
>   journal={Quantum},
>   volume={7},
>   pages={892},
>   year={2023},
>   publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften}
> }
> ```
## Test Instances
There are three test instances in the paper:
* Energy minimization problem
* CNOT gate estimation problem
* Circuit compilation problem

For each instance, we solve the following optimization problems:
* Continuous relaxation
* Model with TV regularizer
* Rounding techniques
* Improvement heuristic

The instances are provided in the folder ```/example/```.

## Installation
### Requirements
* Python >= 3.8
* qiskit >= 0.29.0, scipy >= 1.6.2
* pycombina >= 0.3.2 (only for rounding)
* Developed version of Qutip (https://github.com/qutip/qutip). 

To install pycombina, please refer to https://pycombina.readthedocs.io/en/latest/. 

To install the developed version of Qutip, first clone the 
repository locally:

```shell 
git clone https://github.com/xinyufei/Quantum-Control-qutip.git
```

Then, install Qutip by 

```shell
python setup.py develop
```

## Usage
### Stored results
All the control results are stored in the folder ```example/control/```. All the output control figures are stored in 
```example/figure/```. The output files are stored in ```example/output/```. One can change the 
paths in files to change the positions. 

**Before starting your own experiments, we suggest deleting the above three folders to clear all the existing results.** 
### Continuous relaxation
First, change to the example file folder:
```shell
cd example/Continuous/
```
To run an energy minimization problem with 4 qubits, randomly generated graph for Hamiltonian controllers, 
evolution time as 2, time steps as 40, constant initial control values 0.5, run:
```shell 
python energy.py --n=4 --num_edges=2 --rgraph=1 --seed=1 \
    --evo_time=2 --n_ts=40 --initial_type=CONSTANT --offset=0.5
```
To run a CNOT estimation problem with evolution time 10, time steps 200, constant initial control values 0.5, run:
```shell
python CNOT.py --evo_time=10 --n_ts=200 --initial_type=CONSTANT --offset=0.5
```
To run a circuit compilation problem on molecule LiH with evolution time 20, time steps 200, automatically generating 
target circuit, constant initial control values 0.5, and without SOS1 property, run
```shell
python Molecule.py --gen_target=1 --name=MoleculeVQE \
    --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=100 \
    --initial_type=CONSTANT --offset=0.5
```
In circuit compilation problem, to penalize the SOS1 property, one can set the penalty parameter. For example, with the 
above setting adding penalty, run:
```shell
python Molecule.py --gen_target=1 --name=MoleculeVQE \
    --molecule=LiH --qubit_num=4 --evo_time=20 --n_ts=100 \
    --initial_type=CONSTANT --offset=0.5 --sum_penalty=0.1
```
The output control files are stored in ```example/control/Continuous/```. The output control figures are stored in 
```example/figure/Continuous/```. The output files are stored in ```example/output/Continuous/```. 
### TV regularizer
First, change to the example file folder:
```shell
cd example/ADMM/
```
To run ADMM for solving the model with TV regularizer on energy minimization problem, run:
```shell
python energy.py --n=4 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 --initial_type=WARM \
    --initial_control="../control/Continuous/Energy4_evotime2.0_n_ts240_ptypeCONSTANT_offset0.5_instance1.csv" \
    --alpha=1e-2 --rho=10 --max_iter_admm=50
```
### Rounding
First, change to the example file folder:
```shell
cd example/Rounding/
```
To run SUR on energy minimization problem, run:
```shell 
python energy.py --n=4 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
    --initial_control="../control/Continuous/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
    --type=SUR
```
To run CIA with min-up time constraints on energy minimization problem, run:
```shell
python energy.py --n=4 --rgraph=1 --seed=1 --num_edges=2 --evo_time=2 --n_ts=40 \
    --initial_control="../control/Continuous/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv" \
    --type=minup --min_up=10
```
### Improvement heuristic
First, change to the example file folder:
```shell
cd example/Trustregion/
```
To run the improvement heuristic on energy minimization problem for improve the solutions 
with TV regularizer, run:
```shell
python energy.py --n=4 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
    --initial_file="../control/Rounding/EnergyADMM4_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4_1_SUR.csv" \
    --alpha=0.01 --tr_type="tv"
```
To run the improvement heuristic on energy minimization problem for improve the solutions 
with min-up time constraints, run:
```shell
python energy.py --n=4 --num_edges=2 --rgraph=1 --seed=4 --evo_time=2 --n_ts=40 \
    --initial_file="../control/Rounding/EnergyADMM4_evotime2.0_n_ts40_ptypeWARM_offset0.5_penalty0.01_ADMM_10.0_iter100_instance4_minup10_1.csv" \
    --alpha=0.01 --tr_type="hard" --hard_type="minup" --min_up=10
```

## Acknowledgement
We thank Dr.Lucas Brady for providing the code used in the paper **Optimal Protocols in Quantum Annealing and 
QAOA Problems** (https://arxiv.org/pdf/2003.08952.pdf).

We refer to the paper **Partial Compilation of Variational Algorithms for 
Noisy Intermediate-Scale Quantum Machines** (https://arxiv.org/pdf/1909.07522.pdf) for generating the 
circuit compilation problem. Their code is presented at https://github.com/epiqc/PartialCompilation.