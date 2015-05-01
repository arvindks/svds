# svds
Possible replacement for scipy svds

Scipy sparse.linalg.svds is a computation of the singular value decomposition of A, by a call to ARPACK on the matrix A'*A or A*A' depending on the size of the matrices. However, this computation can be unstable. An alternative is to compute the eigenvalues of the matrix [ 0 A; A' 0] which has eigendecomposition $\frac{1}{2} [\pm u ]$
