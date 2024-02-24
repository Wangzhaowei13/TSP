"""
获取2比特海森堡模型的密度矩阵随时间的演化，并将其转化为概率分布
8.30：将密度矩阵用概率分布表示之后，重新通过态层析的方法重构出密度矩阵
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import qutip
from qutip import basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, destroy, Qobj


# Set the system parameters
N = 1

# initial state
# state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
# state_list = 1 / np.sqrt(2) * ((basis(2, 0) - sigmay() * basis(2, 0)) + (basis(2, 0) - sigmax() * basis(2, 0))) # 单比特
state_list = [1 / np.sqrt(2) * (basis(2, 0) + basis(2, 1))] * N  # N比特
psi0 = tensor(state_list)
# print(psi0)

# Energy splitting term
h = 1 * np.pi * np.ones(N)
# print(h)

# dephasing rate
gamma = 0.5 * np.ones(N)

# Interaction coefficients
Jx = 2 * gamma * np.ones(N)
Jy = 0 * gamma * np.ones(N)
Jz = 1 * gamma * np.ones(N)

# Setup operators for indibidual qubits
sx_list, sy_list, sz_list = [], [], []
c_ops_lsit = []
c_o = [[0, 0], [1, 0]]
# c_o = sigmax()
c_op = Qobj(c_o)   # sigma-
print(c_op)

for i in range(N):
    op_list = [qeye(2)] * N
    op_list[i] = sigmax()
    sx_list.append(tensor(op_list))
    op_list[i] = sigmay()
    sy_list.append(tensor(op_list))
    op_list[i] = sigmaz()
    sz_list.append(tensor(op_list))
    op_list[i] = c_op
    c_ops_lsit.append(tensor(op_list))


# Hamiltonian - Energy splitting terms
H = 0
for i in range(N):
    H -= 0.5 * h[i] * sz_list[i]

# Interaction terms
for n in range(N - 1):
    H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
    H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
    H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]



times = np.linspace(0, 40, 7200)
# times = np.linspace(0, 15, 1200)

# collapse operators
# c_ops = [np.sqrt(gamma[i]) * sz_list[i] for i in range(N)]
c_ops = [gamma[i] * c_ops_lsit[i] for i in range(N)]
# print(c_ops)

# evolution
result = mesolve(H, psi0, times, c_ops, [])

# 获得演化的密度矩阵
rho_list = result.states
# print(rho_list[2])

"""~~~~~~~~~~~~~~将sigmz的期望值随时间的演化画图~~~~~~~~~~~~~~~~~~~"""
# Expectation value 求期望值sigmz，
exp_sz_dephase = expect(sz_list, result.states)
exp_sx_dephase = expect(sx_list, result.states)
exp_sy_dephase = expect(sy_list, result.states)
sz_biaoqian = exp_sz_dephase[0].tolist()
sx_biaoqian = exp_sx_dephase[0].tolist()
sy_biaoqian = exp_sy_dephase[0].tolist()
# print(sz_biaoqian)

# Plot the expecation value
plt.plot(times, exp_sz_dephase[0], label=r"$\langle \sigma_z^{0} \rangle$")
plt.plot(times, exp_sx_dephase[0], label=r"$\langle \sigma_x^{0} \rangle$")
plt.plot(times, exp_sy_dephase[0], label=r"$\langle \sigma_y^{0} \rangle$")

np.save("exp_sigmax", sx_biaoqian)
np.save("exp_sigmay", sy_biaoqian)
np.save("exp_sigmaz", sz_biaoqian)


plt.legend()
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.title("Dynamics of spin chain with qubit dephasing")
plt.show()


"""~~~~~~~~~~~~~~~~~IC - POVM 测量基~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
# 选择正四面体POVM测量基

M1 = (1 / 4) * (qeye(2) + 0 * sigmax() + 0 * sigmay() + 1 * sigmaz())
M2 = (1 / 4) * (qeye(2) + (2 * np.sqrt(2) / 3) * sigmax() + 0 * sigmay() + (-1 / 3) * sigmaz())
M3 = (1 / 4) * (qeye(2) + (-np.sqrt(2) / 3) * sigmax() + np.sqrt(2 / 3) * sigmay() + (-1 / 3) * sigmaz())
M4 = (1 / 4) * (qeye(2) + (-np.sqrt(2) / 3) * sigmax() + (-np.sqrt(2 / 3)) * sigmay() + (-1 / 3) * sigmaz())
M = [M1, M2, M3, M4]
# print("IC - POVM测量算符测量基为：\n", M)

M_list = []
for i in range(N):
    op_list = [qeye(2)] * N
    op_list[i] = M1
    M_list.append(tensor(op_list))
    op_list[i] = M2
    M_list.append(tensor(op_list))
    op_list[i] = M3
    M_list.append(tensor(op_list))
    op_list[i] = M4
    M_list.append(tensor(op_list))

"""~~~~~~~~~~~~~~~~将密度矩阵转换成概率分布~~~~~~~~~~~~~~~~~~~"""

P_all_time_list = []
for t in range(len(times)):
    P_list = []
    for i in M_list:
        P_list.append(np.abs(Qobj.tr(rho_list[t] * i))) # 对tr出来的值要取复数的模
    P_all_time_list.append(P_list)
print(P_all_time_list)
np.save('P_all_time_list.npy',P_all_time_list)




