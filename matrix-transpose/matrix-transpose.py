import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    n = len(A)
    m = len(A[0])
    ans = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            ans[i, j] = A[j][i]
    return ans
            
