from svds import svds as new_svds
import numpy as np
from scipy.sparse import csc_matrix, isspmatrix
from scipy.linalg import svd
from scipy.sparse.linalg import svds as old_svds


def sorted_svd(m, k, which='LM'):
    # Compute svd of a dense matrix m, and return singular vectors/values
    # sorted.
    if isspmatrix(m):
        m = m.todense()
    u, s, vh = svd(m)
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError("unknown which=%r" % (which,))
    return u[:, ii], s[ii], vh[ii]


def svd_estimate(u, s, vh):
    return np.dot(u, np.dot(np.diag(s), vh))

x = np.array([[1, 2, 3],
              [3, 4, 3],
              [1 + 1j, 0, 2],
              [0, 0, 1]], np.complex)
y = np.array([[1, 2, 3, 8 + 5j],
              [3 - 2j, 4, 3, 5],
              [1, 0, 2, 3],
              [0, 0, 1, 0]], np.complex)
z = csc_matrix(x)

m = x
k = 1
u, s, vh = sorted_svd(m, k)
su, ss, svh = old_svds(m, k)
ssu, sss, ssvh = new_svds(m, k)

m_hat = svd_estimate(u, s, vh)
sm_hat = svd_estimate(su, ss, svh)
ssm_hat = svd_estimate(ssu, sss, ssvh)
