from __future__ import division
from cmath import e
import numpy as np
from scipy import *
from sympy.physics.quantum.dagger import Dagger
from scipy.optimize import minimize

n1 = 34749
n2 = 324
n3 = 35805
n4 = 444
n5 = 16324
n6 = 17521
n7 = 13441
n8 = 16901
n9 = 17932
n10 = 32028
n11 = 15132
n12 = 17238
n13 = 13171
n14 = 17170
n15 = 16722
n16 = 33586
n = np.array([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16])
N = n1 + n2 + n3 + n4
# print(N)

m1 = 0.5*np.array([[2.0, -(1 - complex(0,1)), -(1.0 + complex(0,1)), 1], [-(1 + complex(0,1)), 0, complex(0,1), 
     0], [-(1 - complex(0,1)), -complex(0,1), 0, 0], [1, 0, 0, 0]])

m2 = 0.5*np.array([[0, -(1 - complex(0,1)), 0, 1], [-(1 + complex(0,1)), 2, complex(0,1), -(1 + complex(0,1))], [0, -complex(0,1), 0, 
     0], [1, -(1 + complex(0,1)), 0, 0]])

m3 = 0.5*np.array([[0, 0, 0, 1], [0, 0, complex(0,1), -(1 + complex(0,1))], [0, -complex(0,1), 
     0, -(1 - complex(0,1))], [1, -(1 - complex(0,1)), -(1 + complex(0,1)), 2]])

m4 = 0.5*np.array([[0, 0, -(1 + complex(0,1)), 1], [0, 0, complex(0,1), 0], [-(1 - complex(0,1)), -complex(0,1), 
     2, -(1 - complex(0,1))], [1, 0, -(1 + complex(0,1)), 0]])

m5 = 0.5*np.array([[0, 0, 2*complex(0,1), -(1 + complex(0,1))], [0, 0, (1 - complex(0,1)), 0], [-2*complex(0,1), (1 + complex(0,1)), 
     0, 0], [-(1 - complex(0,1)), 0, 0, 0]])

m6 = 0.5*np.array([[0, 0, 0, -(1 + complex(0,1))], [0, 0, (1 - complex(0,1)), 2*complex(0,1)], [0, (1 + complex(0,1)), 0, 
     0], [-(1 - complex(0,1)), -2*complex(0,1), 0, 0]])

m7 = 0.5*np.array([[0, 0, 0, -(1 + complex(0,1))], [0, 0, -(1 - complex(0,1)), 2], [0, -(1 + complex(0,1)), 0, 
     0], [-(1 - complex(0,1)), 2, 0, 0]])

m8 = 0.5*np.array([[0, 0, 2, -(1 + complex(0,1))], [0, 0, -(1 - complex(0,1)), 0], [2, -(1 + complex(0,1)), 0, 
     0], [-(1 - complex(0,1)), 0, 0, 0]])

m9 = np.array([[0, 0, 0, complex(0,1)], [0, 0, -complex(0,1), 0], [0, complex(0,1), 0, 0], [-complex(0,1), 0, 0, 0]])

m10 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

m11 = np.array([[0, 0, 0, complex(0,1)], [0, 0, complex(0,1), 0], [0, -complex(0,1), 0, 0], [-complex(0,1), 0, 0, 0]])

m12 = 0.5*np.array([[0, 2, 0, -(1 + complex(0,1))], [2, 0, -(1 + complex(0,1)), 0], [0, -(1 - complex(0,1)), 0, 
     0], [-(1 - complex(0,1)), 0, 0, 0]])

m13 = 0.5*np.array([[0, 0, 0, -(1 + complex(0,1))], [0, 0, -(1 + complex(0,1)), 0], [0, -(1 - complex(0,1)), 0, 
     2], [-(1 - complex(0,1)), 0, 2, 0]])

m14 = 0.5*np.array([[0, 0, 0, -(1 - complex(0,1))], [0, 0, -(1 - complex(0,1)), 0], [0, -(1 + complex(0,1)), 
     0, -2*complex(0,1)], [-(1 + complex(0,1)), 0, 2*complex(0,1), 0]])

m15 = 0.5*np.array([[0, -2*complex(0,1), 0, -(1 - complex(0,1))], [2*complex(0,1), 0, (1 - complex(0,1)), 0], [0, (1 + complex(0,1)), 
     0, 0], [-(1 + complex(0,1)), 0, 0, 0]])

m16 = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])

M = np.array([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16])

# print(M)

sum = 0
for m in range(16):
    prod = M[m]*n[m]
    sum = sum + prod
rho1 = sum/N
# print("Tomographic reconstruct density matrix  \n", rho1)
# print(np.squeeze(rho1))

d = np.linalg.det(rho1)
# print(d)

# M11 = rho1[1][1]*(rho1[2][2]*rho1[3][3] - rho1[2][3]*rho1[3][2]) - rho1[1][2]*(rho1[2][1]*rho1[3][3] - rho1[2][3]*rho1[3][1]) + rho1[1][3]*(rho1[2][1]*rho1[3][2] - rho1[2][2]*rho1[3][1])
M11 = -0.0502056 - 0.00040892* complex(0,1)

M12 = rho1[1][0]*(rho1[2][2]*rho1[3][3] - rho1[2][3]*rho1[3][2]) - rho1[1][2]*(rho1[2][0]*rho1[3][3] - rho1[2][3]*rho1[3][0]) + rho1[1][3]*(rho1[2][0]*rho1[3][2] - rho1[2][2]*rho1[3][0])
# print(M12)

M1122 = (rho1[2][2]*rho1[3][3] - rho1[2][3]*rho1[3][2])
# print(M1122)

M1223 = (rho1[2][0]*rho1[3][3] - rho1[2][3]*rho1[3][0])
# print(M1223)
M1123 = (rho1[2][1]*rho1[3][3] - rho1[2][3]*rho1[3][1])
# print(M1123)
R41 = rho1[3][0]
# print(R41)
R42 = rho1[3][1]
# print(R42)
R43 = rho1[3][2]
# print(R43)
R44 = rho1[3][3]
# print(R44)

Tin = np.array([[(d/M11)**0.5, 0, 0, 0], [M12/(M11*M1122)**0.5, (M11/M1122)**0.5, 0, 0], [M1223/(R44*M1122)**0.5, M1123/(R44*M1122)**0.5, (M1122/R44)**0.5, 0], [R41/R44**0.5, R42/R44**0.5, R43/R44**0.5, R44**0.5]])
# print(Tin)

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

Tq = np.array([[ta1, 0, 0, 0], [ta5 + complex(0,1)*ta6, ta2, 0, 0], [ta11 + complex(0,1)*ta12, ta7 + complex(0,1)*ta8, ta3, 0], [ta15 + complex(0,1)*ta16, ta13 + complex(0,1)*ta14, ta9 + complex(0,1)*ta10, ta4]])
# print(Tq)

# rho = np.matmul(np.conj(Tq).T,Tq)/np.trace(np.matmul(np.conj(Tq).T,Tq))
rho = np.matmul(Dagger(Tq),Tq)/np.trace(np.matmul(np.conj(Tq).T,Tq))
# print(rho)


H1 = np.array([[1.0],[0.0]])
V1 = np.array([[0.0],[1.0]])
D1 = np.array([[1.0/2**0.5],[1.0/2**0.5]])
R1 = np.array([[1.0/2**0.5],[1.0* complex(0,1)/2**0.5]])
L1 = np.array([[1.0/2**0.5],[1.0* complex(0,-1)/2**0.5]])

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
# print(ta)


# t = np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16])

def L_f(t):
     Tq = np.array([[t[0], 0, 0, 0], [t[4] + complex(0,1)*t[5], t[1], 0, 0], [t[10] + complex(0,1)*t[11], t[6] + complex(0,1)*t[7], t[2], 0], [t[14] + complex(0,1)*t[15], t[12] + complex(0,1)*t[13], t[8] + complex(0,1)*t[9], t[3]]])
     rho = np.matmul(Dagger(Tq),Tq)/np.trace(np.matmul(np.conj(Tq).T,Tq))
     sum1 = 0
     for m in range(16):
          Num = (N*np.squeeze(np.inner(np.conj(state[m]).T,np.squeeze(np.matmul(rho,state[m])))) - n[m])**2
          Deno = 2*N*np.squeeze(np.inner(np.conj(state[m]).T,np.squeeze(np.matmul(rho,state[m]))))* - n[m]
          res = Num/Deno
          sum1 = sum1 + res
     # L = sum/N
     return abs(sum1)

# t_1 = minimize(L_f,ta, method='BFGS')
result = minimize(L_f,ta, method='BFGS', tol=1e-5)
# print(result)
t_min = result.x
# print("Length of t_min array \n", len(t_min))
# print("t_min as a array \n", t_min)

# print("Length of t_min array \n", len(t_min[6]))
# print("t_min as a array \n", t_min[6])

# t_min = np.asarray(t_1)
# print(t_min)

Tqf = np.array([[t_min[0], 0, 0, 0], [t_min[4] + complex(0,1)*t_min[5], t_min[1], 0, 0], [t_min[10] + complex(0,1)*t_min[11], t_min[6] + complex(0,1)*t_min[7], t_min[2], 0], [t_min[14] + complex(0,1)*t_min[15], t_min[12] + complex(0,1)*t_min[13], t_min[8] + complex(0,1)*t_min[9], t_min[3]]])
rho_physical = np.matmul(Dagger(Tqf),Tqf)/np.trace(np.matmul(np.conj(Tqf).T,Tqf))
print("Final density matrix after MLE \n \n", rho_physical)
print("Trace of final density matrix obtained from MLE \n \n", np.trace(rho_physical))
# rho =  np.outer(HL), np.conj(HL))
# Bmat1c11 = np.trace(rho*ga14

