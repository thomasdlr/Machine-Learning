# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:27:03 2021

@author: thoma
"""

a=1


def transpose(A): 
    """
    :param A: matrix to transpose
    :return T: T is the transposed matrix of A
    """
    n=len(A)
    m=len(A[0])
    T=[]
    for i in range(m):
        l=[]
        for j in range(n):
            l.append(A[j][i])
        T.append(l)
    return T


def zeros(n,m):
    """
    Creates a matrix of size n*m filled with null values
    :param n: nb of lines of the created matrix
    :param m: nb of columns of the created matrix
    :return M: null matrix
    """
    M=[]
    for i in range(n):
        L=[]
        for j in range(m):
            L.append(0)
        M.append(L)
    return M



def multiplication(A,B): 
    """
    Performs a matrix multiplication
    :param A: matrix, left element of the matrix multiplication
    :param B: matrix, right element of the matrix multiplication
    :return C: matrix, C=AB
    """
    n, l, p, m = len(A), len(B), len(A[0]), len(B[0])
    if p != l:
        raise ValueError("the nb of columns of A must be equal to the nb of lines of B")
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k]*B[k][j]
    return C


def to_be_inversed(A, l):
    """
    Performs the matrix operation : At A + lI
    :param A: matrix
    :param l: real coefficient
    :return C: matrix
    """
    B = transpose(A)
    n = len(A[0])
    C = multiplication(B, A)
    for i in range(n):
        C[i][i] += l
    return C

def substraction(A, B):
    """
    Performs a matrix substraction
    :param A: matrix
    :param B: matrix
    :return C: matrix
    """
    m = len(A)
    n = len(A[0])
    C = zeros(m, n)
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def LU(M):
    """
    Performs the LU decomposition of a matrix
    :param M: square matrix of size n
    :return [L, U]: list of two square matrices of size n
    """
    n=len(M)
    A=copy.deepcopy(M)
    B=zeros(n,n)
    L=zeros(n,n)
    U=zeros(n,n)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                B[i][j]=A[i][k]/A[k][k]*A[k][j]
                L[i][k]=A[i][k]/A[k][k]
                U[k][j]=A[k][j]
        B=substraction(A,B)
        A=copy.deepcopy(B)
    return[L,U]


def inverse_trilow(A):
    """
    Gauss method to inverse a lower triangular matrix
    :param A: lower triangular matrix
    :return Inv: matrix, Inv = A**(-1)
    """
    n = len(A)
    T = copy.deepcopy(A)
    Inv = zeros(n,n)
    for i in range(n):
        Inv[i][i] = 1
    for k in range(n-1):
        pivot = T[k][k]
        for i in range(k+1, n):
            for j in range(n):
                T[i][j] = T[i][j] - T[i][k] / pivot * T[k][j]
                Inv[i][j] = Inv[i][j]-T[i][k]/pivot*Inv[k][j]
    for k in range(n):
        if T[k][k] != 1:
            c = T[k][k]
            for j in range(n):
                Inv[k][j] = Inv[k][j] / c
                T[k][j] = T[k][j]/c
    return Inv


def inverseLU(A):
    """
    use of LU decomposition to inverse a matrix A
    :param A: matrix
    :return Inv: matrix, Inv = A**(-1)
    """
    [L,U] = LU(A)
    Ut = transpose(U)
    Li = inverse_trilow(L)
    Uti = inverse_trilow(Ut)
    Ui = transpose(Uti)
    Inv = multiplication(Ui, Li)
    return Inv


def different(A,B):
    """
    Tells if two matrices are identical
    :param A: matrix
    :param B: matrix
    :return res: boolean
    """
    m = len(A)
    n = len(A[0])
    res = False
    for i in range(m):
        for j in range(n):
            if A[i][j] == B[i][j]:
                res = True
    return res

def scalar_multi(l,M):
    """
    Multiplies a matrix by a scalar
    :param l: real coefficient
    :param M: Matrix
    :return N: l*M
    """
    N = M.copy()
    m = len(M)
    n = len(M[0])
    for i in range(m):
        for j in range(n):
            N[i][j] = l*M[i][j]
    return N
  
  
  