import numpy as np
from scipy.special import comb, factorial

def det(m_max, n_max, eta):
    det = np.zeros((m_max + 1, n_max + 1))
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if m > n:
                det[m, n] = 0
            elif m < n:
                summary = [((-1) ** j) * comb(m, j) * ((1 - eta) + ((m - j) * eta) / m_max) ** n
                           for j in range(m + 1)]
                det[m, n] = comb(m_max, m) * np.sum(summary)
            else:
                det[m, n] = (eta / m_max) ** n * (factorial(m_max) / factorial(m_max - n))
    return det

def eme(m_max, n_max, eta, det, l, c):
    iterations = int(1e10)
    epsilon = 1e-12
    pn = np.array([1 / (n_max + 1)] * (n_max + 1))
    for c_idx, c_value in enumerate(c):
        for i in range(iterations):
            em = np.dot(c_value / np.dot(det, pn), det) * pn
            e = l * (np.log(pn) - np.sum(pn * np.log(pn))) * pn
            e[np.isnan(e)] = 0.0
            eme = em - e
            dist = np.sqrt(np.sum((eme - pn) ** 2))
            if dist <= epsilon:
                break
            else:
                pn = eme
    return eme

m_max = 10
n_max = 50
eta = 0.5
l = 1e-3
c = np.array([6.73794700e-03, 4.37104954e-02, 1.27601677e-01, 2.20741125e-01,
              2.50599060e-01, 1.95082729e-01, 1.05461930e-01, 3.90945126e-02,
              9.51054071e-03, 1.37104223e-03, 8.89424261e-05])

det_mat = det(m_max, n_max, eta)
p = eme(m_max, n_max, eta, det_mat, l, c)
print(p)
