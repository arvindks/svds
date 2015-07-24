from svds import svds as new_svds
import numpy as np
from scipy.sparse import csc_matrix, isspmatrix
from scipy.linalg import svd, hilbert
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


n = 5
x = hilbert(n)
# u,s,vh = svd(x)
# s[2:-1] = 0
# x = svd_estimate(u, s, vh)

which = 'SM'
k = 2
u, s, vh = sorted_svd(x, k, which=which)
su,ss,svh = old_svds(x, k, which=which)
ssu,sss,ssvh = new_svds(x, k, which=which)

m_hat = svd_estimate(u, s, vh)
sm_hat = svd_estimate(su, ss, svh)
ssm_hat = svd_estimate(ssu, sss, ssvh)