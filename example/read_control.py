import numpy as np

N = 5
T = 100
additional_idx = [((n + 1) * (T + 1) - 1) * 2 for n in range(N)]
ctrl = np.zeros(N * T)
f = open("control/Continuous/MoleculeH2_evotime10_nts_100_bonmin.csv")
lines = f.readlines()
new_lines = []
num = 0
for line_idx, line in enumerate(lines):
    # print(line)
    line = line.strip('\n')
    if len(line) > 0:
        if line_idx not in additional_idx:
            ctrl[num] = float(line.split(" = ")[-1])
            num += 1
    new_lines.append(line)
    # print(line)
print(ctrl)
reshaped_ctrl = ctrl.reshape((N, T))
print(reshaped_ctrl)
tv = sum(sum(abs(reshaped_ctrl[j, t] - reshaped_ctrl[j, t - 1]) for t in range(1, T)) for j in range(N))
print(tv)
# print(ctrl.reshape((N, T)))