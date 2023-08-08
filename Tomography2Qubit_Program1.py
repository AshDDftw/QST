from __future__ import division
import numpy as np
from scipy.optimize import minimize
from sympy.physics.quantum.dagger import Dagger
from cmath import e
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = [34749, 324, 35805, 444, 16324, 17521, 13441, 16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586]
N = sum(n)


import numpy as np

m1 = 0.5 * np.array([[2, -1 + 1j, -1 - 1j, 1],
                     [-1 - 1j, 0, 1j, 0],
                     [-1 + 1j, -1j, 0, 0],
                     [1, 0, 0, 0]])

m2 = 0.5 * np.array([[0, -1 + 1j, 0, 1],
                     [-1 - 1j, 2, 1j, -1 - 1j],
                     [0, -1j, 0, 0],
                     [1, -1 - 1j, 0, 0]])

m3 = 0.5 * np.array([[0, 0, 0, 1],
                     [0, 0, 1j, -1 - 1j],
                     [0, -1j, 0, -1 + 1j],
                     [1, -1 + 1j, -1 - 1j, 2]])

m4 = 0.5 * np.array([[0, 0, -1 - 1j, 1],
                     [0, 0, 1j, 0],
                     [-1 + 1j, -1j, 2, -1 + 1j],
                     [1, 0, -1 - 1j, 0]])

m5 = 0.5 * np.array([[0, 0, 2j, -1 - 1j],
                     [0, 0, 1 - 1j, 0],
                     [-2j, 1 + 1j, 0, 0],
                     [-1 + 1j, 0, 0, 0]])

m6 = 0.5 * np.array([[0, 0, 0, -1 - 1j],
                     [0, 0, 1 - 1j, 2j],
                     [0, 1 + 1j, 0, 0],
                     [-1 + 1j, -2j, 0, 0]])

m7 = 0.5 * np.array([[0, 0, 0, -1 - 1j],
                     [0, 0, -1 + 1j, 2],
                     [0, -1 - 1j, 0, 0],
                     [-1 + 1j, 2, 0, 0]])

m8 = 0.5 * np.array([[0, 0, 2, -1 - 1j],
                     [0, 0, -1 + 1j, 0],
                     [2, -1 - 1j, 0, 0],
                     [-1 + 1j, 0, 0, 0]])

m9 = np.array([[0, 0, 0, 1j],
               [0, 0, -1j, 0],
               [0, 1j, 0, 0],
               [-1j, 0, 0, 0]])

m10 = np.array([[0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0]])

m11=np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]])

m12 = 0.5*np.array([[0, 2, 0, -(1 + 1j)], [2, 0, -(1 + 1j), 0], [0, -(1 - 1j), 0, 0], [-(1 - 1j), 0, 0, 0]])

m13 = 0.5*np.array([[0, 0, 0, -(1 + 1j)], [0, 0, -(1 + 1j), 0], [0, -(1 - 1j), 0,2], [-(1 - 1j), 0, 2, 0]])

m14 = 0.5*np.array([[0, 0, 0, -(1 - 1j)], [0, 0, -(1 - 1j), 0], [0, -(1 + 1j),0, -2*1j], [-(1 + 1j), 0, 2*1j, 0]])

m15 = 0.5*np.array([[0, -2*1j, 0, -(1 - 1j)], [2*1j, 0, (1 - 1j), 0], [0, (1 + 1j),0, 0], [-(1 + 1j), 0, 0, 0]])

m16 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])

M = np.array([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16])

sum = 0
for m in range(16):
    prod = M[m]*n[m]
    sum = sum + prod
rho1 = sum/N
d = np.linalg.det(rho1)

M11 = -0.0502056 - 0.00040892j

M12 = rho1[1][0]*(rho1[2][2]*rho1[3][3] - rho1[2][3]*rho1[3][2]) - rho1[1][2]*(rho1[2][0]*rho1[3][3] - rho1[2][3]*rho1[3][0]) + rho1[1][3]*(rho1[2][0]*rho1[3][2] - rho1[2][2]*rho1[3][0])
M1122 = rho1[2][2]*rho1[3][3] - rho1[2][3]*rho1[3][2]
M1223 = rho1[2][0]*rho1[3][3] - rho1[2][3]*rho1[3][0]
M1123 = rho1[2][1]*rho1[3][3] - rho1[2][3]*rho1[3][1]

R4 = rho1[3]

R41 = rho1[3][0]
# print(R41)
R42 = rho1[3][1]
# print(R42)
R43 = rho1[3][2]
# print(R43)
R44 = rho1[3][3]

# # Define the numerator and denominator terms
# num1 = d
# num2 = -0.0502056 - 0.00040892* complex(0,1)
# denom1 = M11
# denom2 = M1122

# # Calculate the matrix elements
# in11 = (num1 / denom1) ** 0.5
# in12 = M12 / (denom1 * denom2) ** 0.5
# in13 = M1223 / (R44 * denom2) ** 0.5
# in14 = R41 / R44 ** 0.5

# in21 = (denom1 / denom2) ** 0.5
# in22 = (num1 / denom2 - in12**2) ** 0.5
# in23 = M1123 / (R44 * denom2) ** 0.5
# in24 = R42 / R44 ** 0.5

# in31 = 0
# in32 = 0
# in33 = (denom2 / R44) ** 0.5
# in34 = R43 / R44 ** 0.5

# in41 = 0
# in42 = 0
# in43 = 0
# in44 = R44 ** 0.5

# Assemble the matrix
# Tq = np.array([[in11, in12, in13, in14],
#                [in21, in22, in23, in24],
#                [in31, in32, in33, in34],
#                [in41, in42, in43, in44]])


ta1 = 0.0024076123578187884
ta2 = 5.171099978738858
ta3 = 3.74470335725104*10**-18
ta4 = 0.7085330001957924
ta5 = 0.6723831111390692
ta6 = -0.7208511874915722
ta7 = -3.6626133561171654
ta8 = 3.638177180769397
ta9 = -0.09802313130763464
ta10 = -0.018888276740312356
ta11 = -0.020606220573770387
ta12 = -0.99852641726235
ta13 = -0.09149287324131397
ta14 = 0.004343611047143596
ta15 = 0.732793898345214
ta16 = -0.05366684810867167

t0 = np.array([[ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8, ta9, ta10, ta11, ta12, ta13, ta14, ta15, ta16]])

Tq = np.array([[ta1, 0, 0, 0], [ta5 + 1j*ta6, ta2, 0, 0], [ta11 + 1j*ta12, ta7 + 1j*ta8, ta3, 0], [ta15 + 1j*ta16, ta13 + 1j*ta14, ta9 + 1j*ta10, ta4]])

# Compute density matrix from Tq
rho = np.matmul(np.conj(Tq).T, Tq) / np.trace(np.matmul(np.conj(Tq).T, Tq))

# Define polarization states
H1 = np.array([[1.0], [0.0]])
V1 = np.array([[0.0], [1.0]])
D1 = np.array([[1.0/2**0.5], [1.0/2**0.5]])
R1 = np.array([[1.0/2**0.5], [1.0*complex(0, 1)/2**0.5]])
L1 = np.array([[1.0/2**0.5], [1.0*complex(0, -1)/2**0.5]])

HH = np.kron(H1, H1)
HV = np.kron(H1, V1)
VV = np.kron(V1, V1)
VH = np.kron(V1, H1)
RH = np.kron(R1, H1)
RV = np.kron(R1, V1)
DV = np.kron(D1, V1)
DH = np.kron(D1, H1)
DR = np.kron(D1, R1)
DD = np.kron(D1, D1)
RD = np.kron(R1, D1)
HD = np.kron(H1, D1)
VD = np.kron(V1, D1)
VL = np.kron(V1, L1)
HL = np.kron(H1, L1)
RL = np.kron(R1, L1)

state = np.array([HH, HV, VV, VH, RH, RV, DV, DH, DR, DD, RD, HD, VD, VL, HL, RL])
# state_con = np.array([Dagger(HH), Dagger(HV), Dagger(VV), Dagger(VH), Dagger(RH), Dagger(RV), Dagger(DV), Dagger(DH), Dagger(DR), Dagger(DD), Dagger(RD), Dagger(HD), Dagger(VD), Dagger(VL), Dagger(HL), Dagger(RL)])

ta = np.array([ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8, ta9, ta10, ta11, ta12, ta13, ta14, ta15, ta16])

# ta=np.array([in11, in12, in13, in14,in21, in22, in23, in24,in31, in32, in33, in34,in41, in42, in43, in44])

def L_f(t):
    Tq = np.array([
        [t[0], 0, 0, 0],
        [t[4] + 1j*t[5], t[1], 0, 0],
        [t[10] + 1j*t[11], t[6] + 1j*t[7], t[2], 0],
        [t[14] + 1j*t[15], t[12] + 1j*t[13], t[8] + 1j*t[9], t[3]]
    ])
    rho = np.matmul(Tq.conj().T, Tq) / np.trace(np.matmul(np.conj(Tq).T, Tq))
    
    sum1 = 0
    for m in range(16):
        num = (N * np.abs(np.matmul(state[m].conj().T, np.matmul(rho, state[m]))) - n[m]) ** 2
        den = 2 * N * np.abs(np.matmul(state[m].conj().T, np.matmul(rho, state[m]))) - n[m]
        res = num / den
        sum1 += res
    
    return abs(sum1)

result = minimize(L_f, ta, method='BFGS', tol=1e-5)
print("Success status: ", result.success)
print("Number of function evaluations: ", result.nfev)
print("Minimum value of the function: ", result.fun)
print("Minimizer: ", result.x)


# Construct the final Tq matrix using the optimized parameters
t_min = minimize(L_f, ta, method='BFGS', tol=1e-5).x
Tqf = np.array([[t_min[0], 0, 0, 0], 
                [t_min[4] + complex(0,1)*t_min[5], t_min[1], 0, 0], 
                [t_min[10] + complex(0,1)*t_min[11], t_min[6] + complex(0,1)*t_min[7], t_min[2], 0], 
                [t_min[14] + complex(0,1)*t_min[15], t_min[12] + complex(0,1)*t_min[13], t_min[8] + complex(0,1)*t_min[9], t_min[3]]])

# Obtain the final physical density matrix using the Tq matrix
rho_physical = np.matmul(Dagger(Tqf), Tqf) / np.trace(np.matmul(np.conj(Tqf).T, Tqf))

# Print the final density matrix and its trace
print("Final density matrix after MLE: \n\n", rho_physical)
print("Trace of final density matrix obtained from MLE: \n\n", np.trace(rho_physical))



# Create a complex matrix
# matrix = np.array([[1+2j, 2+3j], [3+4j, 4+5j]])

# Separate the real and imaginary parts of the matrix
real_part = np.real(rho_physical)
imag_part = np.imag(rho_physical)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the real part of the matrix
X = np.arange(real_part.shape[0])
Y = np.arange(real_part.shape[1])
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(real_part)
dx = dy = 0.5
dz = real_part.ravel()
ax.bar3d(X.ravel(), Y.ravel(), Z.ravel(), dx, dy, dz, color='blue')

# Set the labels for the X and Y axes
# xticklabels = ['VH', 'HH','VH','HH']
# yticklabels = ['VV', 'VH','HV','HH']
# ax.set_xticklabels(xticklabels)
# ax.set_yticklabels(yticklabels)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Real')

plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# Plot the imaginary part of the matrix
X = np.arange(imag_part.shape[0])
Y = np.arange(imag_part.shape[1])
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(imag_part)
dz = imag_part.ravel()
ax1.bar3d(X.ravel(), Y.ravel(), Z.ravel(), dx, dy, dz, color='red')

# Set the labels and title
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Imaginary')

plt.show()
