"""
Hamiltonians for different physical qubit systems.
Right now only implements Hamiltonian for SchusterLab transmon qubit.
"""

import numpy as np
from scipy.linalg import expm
from tools.circuitutil import *
from tools.uccsdcircuit import *

# All of these frequencies below are in GHz:
G_MAXA = 2 * np.pi * 0.05
CHARGE_DRIVE_MAXA = 2 * np.pi * 0.1
FLUX_DRIVE_MAXA = 2 * np.pi * 1.5


def get_H0(N, d):
    """Returns the drift Hamiltonian, H0."""
    return np.zeros((d ** N, d ** N))


def _validate_connectivity(N, connected_qubit_pairs):
    """Each edge should be included only once."""
    for (j, k) in connected_qubit_pairs:
        assert 0 <= j < N
        assert 0 <= k < N
        assert j != k
        assert connected_qubit_pairs.count((j, k)) == 1
        assert connected_qubit_pairs.count((k, j)) == 0


def get_Hops_and_Hnames(N, d, connected_qubit_pairs):
    """Returns the control Hamiltonian matrices and their labels."""
    hamiltonians, names = [], []
    for j in range(N):
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s charge drive" % j)

        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) @ get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s flux drive" % j)

    _validate_connectivity(N, connected_qubit_pairs)
    for (j, k) in connected_qubit_pairs:
        matrices = [np.eye(d)] * N
        matrices[j] = get_adagger(d) + get_a(d)
        matrices[k] = get_adagger(d) + get_a(d)
        hamiltonians.append(krons(matrices))
        names.append("qubit %s-%s coupling" % (j, k))

    return hamiltonians, names


def get_maxA(N, d, connected_qubit_pairs):
    """Returns the maximium amplitudes of the control pulses corresponding to Hops/Hnames."""
    maxA = []
    for j in range(N):
        maxA.append(CHARGE_DRIVE_MAXA)  # max amp for charge drive on jth qubit
        maxA.append(FLUX_DRIVE_MAXA)  # max amp for flux drive on jth qubit

    for (j, k) in connected_qubit_pairs:
        maxA.append(G_MAXA)  # max amp for coupling between qubits j and k

    return maxA


def get_a(d):
    """Returns the matrix for the annihilation operator (a^{\dagger}), truncated to d-levels."""
    values = np.sqrt(np.arange(1, d))
    return np.diag(values, 1)


def get_adagger(d):
    """Returns the matrix for the creation operator (a^{\dagger}), truncated to d-levels."""
    return get_a(d).T  # real matrix, so transpose is same as the dagger


def get_number_operator(d):
    """Returns the matrix for the number operator, a^\dagger * a, truncated to d-levels"""
    return get_adagger(d) @ get_a(d)


def krons(matrices):
    """Returns the Kronecker product of the given matrices."""
    result = [1]
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def get_full_states_concerned_list(N, d):
    states_concerned_list = []
    for i in range(2 ** N):
        bits = "{0:b}".format(i)
        states_concerned_list.append(int(bits, d))
    return states_concerned_list


def generate_molecule_func(N, d, molecule, optimize=True):
    connected_qubit_pairs = get_nearest_neighbor_coupling_list(2, int(N / 2), directed=False)
    print(connected_qubit_pairs)
    H0 = get_H0(N, d).astype("complex128")
    Hops, Hnames = get_Hops_and_Hnames(N, d, connected_qubit_pairs)
    states_concerned_list = get_full_states_concerned_list(N, d)
    maxA = get_maxA(N, d, connected_qubit_pairs)

    circuit = get_uccsd_circuit(molecule, optimize=optimize)
    U = get_unitary(circuit).astype("complex128")

    # print(circuit)
    # print(H0.size)
    # print(len(Hops))
    # print(U.size)
    # print(connected_qubit_pairs)
    # for i in range(len(Hops)):
    #     np.savetxt("../hamiltonians/" + molecule + "_controller_" + str(i + 1) + ".csv", Hops[i])
    Hops_new = [Hops[idx].astype("complex128") * maxA[idx] for idx in range(len(Hops))]
    U0 = np.identity(2 ** N).astype("complex128")
    return Hops_new, H0, U0, U


if __name__ == '__main__':
    Hops, H0, U0, U = generate_molecule_func(2, 2, "H2", True)
    print(U)
    print(len(Hops))
    for H in Hops:
        u, s, vh = np.linalg.svd(H, full_matrices=True)
        print(s)
        print(sum(s))

    u, s, vh = np.linalg.svd((Hops[0] + Hops[1])/2, full_matrices=True)
    print(s)
    print(sum(s))

    exit()

    # corresponding = []
    # trace_list = []
    # trace_list_idx = []
    # for i in range(len(Hops)):
    #     for j in range(len(Hops)):
    #         commute = Hops[i].dot(Hops[j]) - Hops[j].dot(Hops[i])
    #         trace = np.trace(U.conj().T.dot(Hops[i].dot(Hops[j])).dot(U0))
    #         # print(commute)
    #         if not (commute == 0).all():
    #             print(commute[commute != 0])
    #             corresponding.append((i + 1, j + 1))
    #         if trace != 0:
    #             trace_list.append(trace)
    #             trace_list_idx.append((i + 1, j + 1))
    # print(corresponding)
    # print(trace_list)
    # print(trace_list_idx)
    # exit()

    Hops, H0, U0, U = generate_molecule_func(6, 2, "BeH2", True)
    control = np.loadtxt(
        "../example/control/ADMM/MoleculeADMM_BeH2_evotime5.0_n_ts50_ptypeWARM_offset0.5_sum_penalty0.01_penalty0.001_ADMM_3.0_iter100_extend_ts_100.csv",
        delimiter=",")
    delta_t = 5 / 100
    n_ts = 100
    X = [U0]
    H_t_0 = np.identity(2 ** 6) - 1j * delta_t * (H0.copy() + sum(control[0, j] * Hops[j].copy() for j in range(19)))
    X_t = H_t_0.dot(X[0])
    X_control = []
    X_control.append(X_t)
    for j in range(19):
        H_t_0 = np.identity(2 ** 6) - 1j * delta_t * (H0.copy() + Hops[j].copy())
        X_t = H_t_0.dot(X[0])
        X_control.append(X_t)
    X_remain = [np.identity(2 ** 6)]
    for t in range(1, n_ts):
        H_t = H0.copy()
        for j in range(19):
            H_t += control[t, j] * Hops[j].copy()
        X_t = expm(-1j * H_t * delta_t).dot(X_remain[t - 1])
        X_remain.append(X_t)
    fid_control = []
    for j in range(20):
        fid = 1 - np.abs(np.trace(
            np.linalg.inv(U).dot(X_remain[-1].dot(X_control[j])))) / U.shape[0]
        fid_control.append(fid)
    print(fid_control)
    print(np.argmin(np.array(fid_control[1:])))
    print(-sum(fid_control[1:] - fid_control[0]))
    fid_control = []
    tau = 0
    temp_H_t = sum(H0.copy() + sum(control[tau, j] * Hops[j].copy() for j in range(19)))
    for j in range(20):
        if j == 0:
            H_t_0 = np.identity(2 ** 6) - 1j * delta_t * temp_H_t
        else:
            H_t_0 = np.identity(2 ** 6) - 1j * delta_t * (
                    temp_H_t - sum(control[0, j] * Hops[j].copy() for j in range(19)) + Hops[j - 1].copy())
        # print([H_t_0.dot(X[0])[i, 0] for i in range(2 ** 6)])
        # print(np.linalg.inv(U))
        # fid = np.abs(np.trace(
        #     np.linalg.inv(U).dot(H_t_0.dot(X[0])))) / U.shape[0]
        fid = np.abs(np.trace(
            np.linalg.inv(U).dot(X_remain[-1].dot(H_t_0.dot(X[0]))))) / U.shape[0]
        fid_control.append(fid)
    print(fid_control)
    print(np.argmax(np.array(fid_control[1:])))
    print([fid_control[i] - fid_control[0] for i in range(19)])
    print(-sum(fid_control[1:] - fid_control[0]))
    exit()
    # for i in range(19):
    #     for j in range(19):
    #         if i != j:
    #             trace = np.trace(
    #                     np.linalg.inv(U).dot(Hops[i].copy().dot(Hops[j].copy()).dot(X[0]))) / U.shape[0]
    #             if np.abs(trace) > 0:
    #                 print(i, j)
    # print((np.linalg.inv(U).dot((Hops[j].copy()).dot(X[0])) == np.zeros(2 ** 6)).all())

    tau = 25

    print([np.trace(U.conj().T.dot(Hops[j]).dot(U0)) for j in range(19)])

    H_t_list = [sum(control[k, j] * Hops[j] for j in range(19)) for k in range(n_ts)]
    # H_t_list = [H_t_1order[k].dot(H_t_1order[k]) for k in range(n_ts)]
    # H_t_list = [Hops[3] for k in range(n_ts)]
    H_t_controller = [H_t_list]
    for j in range(19):
        H_t_controller.append([0] * n_ts)
        for t in range(n_ts):
            H_t_controller[j + 1][t] = H_t_list[t]
        H_t_controller[j + 1][tau] = Hops[j]
    H_t_2 = []
    fid_2 = []
    for j in range(20):
        cur_control = np.zeros((2 ** 6, 2 ** 6), dtype='complex128')
        for k1 in range(n_ts):
            for k2 in range(k1):
                cur_control -= H_t_controller[j][k1].dot(H_t_controller[j][k2])
            for k2 in range(k1 + 1, n_ts):
                cur_control -= H_t_controller[j][k2].dot(H_t_controller[j][k1])
            cur_control += H_t_controller[j][k1].dot(H_t_controller[j][k1]) / 2
        H_t_2.append(cur_control)
        fid_2.append(np.trace(np.linalg.inv(U).dot(H_t_2[j]).dot(X[0])))

    print(fid_2)
    print(sum([np.abs(fid) - np.abs(fid_2[0]) for fid in fid_2[1:]]))
    print(sum(fid - fid_2[0] for fid in fid_2[1:]))

    exit()

    H_t_test = 1j * delta_t * (19 * sum(control[0, j] * Hops[j].copy() for j in range(19)) - sum(Hops[j].copy()
                                                                                                 for j in range(19)))
    print(np.abs(np.trace(np.linalg.inv(U).dot(X_remain[-1].dot(H_t_test.dot(X[0]))))) / U.shape[0])

    H_t_0 = H0.copy() + sum(control[0, j] * Hops[j].copy() for j in range(19))
    X_t = expm(-1j * delta_t * H_t_0).dot(X[0])
    X_control_all = []
    X_control_all.append(X_t)
    for j in range(19):
        H_t_0 = H0.copy() + Hops[j].copy()
        X_t = expm(-1j * delta_t * H_t_0).dot(X[0])
        X_control_all.append(X_t)
    fid_control_all = []
    for j in range(20):
        fid = np.abs(np.trace(
            np.linalg.inv(U).dot(X_remain[-1].dot(X_control_all[j])))) / U.shape[0]
        fid_control_all.append(fid)
    print(fid_control_all)
    print(-sum(fid_control_all[1:] - fid_control_all[0]))
    exit()
    # np.random.seed(0)
    from scipy.stats import unitary_group

    Z = unitary_group.rvs(2 ** 6)
    trace = [sum((Hops[j].dot(Z))[i, i] for i in range(2 ** 6)) for j in range(19)]
    print([np.abs(trace[j]) for j in range(19)])
    # coeffs = np.random.rand(19, 1)
    coeffs = np.array([0] + [1 / 18] * 18)
    sum_of_first = np.sum(coeffs[:18], axis=0)
    coeffs[18] = 1 - sum_of_first  # (*)
    while coeffs[18] <= 0:
        coeffs = np.random.rand(19, 1)
        sum_of_first = np.sum(coeffs[:18], axis=0)
        coeffs[18] = 1 - sum_of_first  # (*)
    print(sum([np.abs(trace[j]) for j in range(19)]))
    print(19 * np.abs(sum(coeffs[j] * trace[j] for j in range(19))))
    print(U.conj().T.dot(U))
