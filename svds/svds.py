import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import aslinearoperator, LinearOperator, eigsh
from scipy.sparse.linalg import svds as svds_
from scipy.sparse.linalg import ArpackNoConvergence


def svds(A, k=6, which='LM', ncv=None, tol=0, v0=None, maxiter=None, return_singular_vectors=True):
    """
    Compute the largest k singular values/vectors for a sparse matrix.
    Parameters
    ----------
    A : {sparse matrix, LinearOperator}
            Array to compute the SVD on, of shape (m, n)
    k : int, optional
            Number of singular values and vectors to compute.
    ncv : int, optional
            The number of Lanczos vectors generated
            ncv must be greater than 2*k+2 and smaller than m + n;
            it is recommended that ncv > 2*k
            Default: ``min(m+n, 4*k + 2)``
    tol : float, optional
            Tolerance for singular values. Zero (default) means machine precision.
    which : str, ['LM' | 'SM'], optional
            Which `k` singular values to find:
            - 'LM' : largest singular values
            - 'SM' : smallest singular values
    v0 : ndarray, optional
            Starting vector for iteration, of length (m+n). 
            Default: random
    maxiter: int, optional
            Maximum number of iterations.
    return_singular_vectors : bool, optional
            Return singular vectors (True) in addition to singular values
    Returns
    -------
    u : ndarray, shape=(m, k)
            Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
            The singular values.
    vt : ndarray, shape=(k, n)
            Unitary matrix having right singular vectors as rows.
    Notes
    -----
    This is a naive implementation using ARPACK as an eigensolver
    on [0 A; A' 0]
    """

    if not (isinstance(A, LinearOperator) or isspmatrix(A)):
        A = aslinearoperator(A)

    m, n = A.shape

    if v0 is not None:
        v0 = np.concatenate((v0, v0))

    if ncv is not None:
        assert ncv > 2 * k + 2 and ncv < m + n,\
            "ncv must be greater than 2*k+2 and smaller than m + n;"

    # Define cyclic matrix operator [0 A; A' 0]
    def matvec(x):
        return np.concatenate((A.matvec(x[n:]), A.rmatvec(x[:n])))

    C = LinearOperator(shape=(m + n, m + n), matvec=matvec, dtype=A.dtype,
                       rmatvec=matvec)

    if return_singular_vectors:
        eigvals, eigvecs = eigsh(C, k=2 * k, which=which, tol=tol,
                                 maxiter=maxiter, ncv=ncv, v0=v0,
                                 return_eigenvectors=return_singular_vectors)

        order = np.argsort(eigvals)[k:]
        eigvals = eigvals[order]
        u = eigvecs[:m, order] / np.sqrt(2)
        v = eigvecs[m:, order] / np.sqrt(2)

        return u, eigvals, v.T

    else:
        eigvals = eigsh(C, k=2 * k, which=which, tol=tol,
                        maxiter=maxiter, ncv=ncv, v0=v0,
                        return_eigenvectors=return_singular_vectors)
        order = np.argsort(eigvals)[k:]
        return eigvals[order]


if __name__ == '__main__':

    S = 0.98**np.arange(100)
    print "True top 6:"
    print np.sort(S[:6])
    print "True bottom 6:"
    print np.sort(S[-6:])

    A = np.diag(S)

    u, s, v = svds(A, k=6, which='LM', return_singular_vectors=True)
    print "New SVDS top 6:"
    print s

    try:
        u, s, v = svds(A, k=6, which='SM', return_singular_vectors=True)
        print "New SVDS bottom 6"
        print s

    except ArpackNoConvergence:
        print "Smallest singular value computation failed"

    s = np.linalg.svd(A, compute_uv=False)
    print "LAPACK top 6:"
    print np.sort(s[:6:])
    print "LAPACK bottom 6:"
    print np.sort(s[-6:])

    u, s, v = svds_(A, k=6, which='LM', return_singular_vectors=True)
    print "Old SVDS top 6"
    print s

    try:
        u, s, v = svds_(A, k=6, which='SM', return_singular_vectors=True)
        print "Old SVDS bottom 6:"
        print s
    except ArpackNoConvergence:
        print "Smallest singular value computation failed"
