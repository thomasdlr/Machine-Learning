import copy
import matplotlib.pyplot as plt
import numpy as np
from matrix_operations import *

"""
NB: In this homework I tried using as few libraries as possible
"""

# a. For LSE
# 1


def read_data(file="testfile.txt", n):
    """
    Stores the data of a file into a matrix and a vector
    :param file: a csv file with 2 columns, X and Y
    :param n: n-1 is the degree that we want to use for the regression
    :return A: matrix, on a given line, there are n columns corresponding to x**i with i from 0 to n-1
    :return B: vector with y values
    """
    A, B = [], []
    fichier = open(file,"r")
    X = fichier.read()
    Y = X.split(' ')
    m = len(Y)
    for i in range(m):
        l = []
        Z = Y[i].split(',')
        a = float(Z[0])
        for k in reversed(range(n)):
            l.append(a**k)
        A.append(l)
        B.append([float(Z[1])])
    return [A, B]


def LSE(file,n,l):
    """
    Finds the X veryfying (AtA)**(-1)*X=AtB
    :param file: a csv file with 2 columns, X and Y
    :param n: n-1 is the degree that we want to use for the regression
    :param l: real coefficient
    :return X: matrix
    :return R: matrix, R=AX
    """
    [A,B] = data(file,n)
    C = to_be_inversed(A,l)
    [L,U] = LU(C)
    m = len(C)
    At = transpose(A)
    D = multiplication(At,B)
    p = len(D[0])
    Y = zeros(m,p) # We define Y=UX and solve LY=D
    for i in range(m):
        for j in range(p):
            Y[i][j] = D[i][j]
            if i != 0:
                for k in range(i):
                    Y[i][j] -= L[i][k]*Y[k][j]
    X = zeros(m,p) #We find X from Y=UX
    for i in reversed(range(m)):
        for j in range(p):
            X[i][j] = 1/U[i][i]*Y[i][j]
            if i != m-1:
                for k in range(i+1,m):
                    X[i][j] -= 1/U[i][i]*U[i][k]*X[k][j]
    R = multiplication(A,X)
    return [X,R]



#a.2

def equation(file, n, l, method):
    """
    Prints the equation of the regression
    :param file: a csv file with 2 columns, X and Y
    :param n: n-1 is the degree that we want to use for the regression
    :param l: real coefficient
    :param method: str, 'LSE' or 'Newton'
    """
    if method == "LSE":
        [X,R] = LSE(file,n,l)
    elif method == "Newton":
        [X,R,c] = Newton(file, n)
    else:
        raise ValueError("Please entre a valid method (LSE or Newton)")
    print('the equation of the best ﬁtting line is :')
    for i in range(n-1):
        print(str(X[i][0]) + ' * X**' + str(n-i-1) + '+')
    print(str(X[n-1][0]))



"""
equation(2,0)   => 4.432950310076807 * X**1+ 29.306404706056256
equation(3,0)   => 3.0238533934865712 * X**2+ 4.906190263863799 * X**1  -0.23140175608771987
equation(3,10000)   => 0.8345332827002858 * X**2+ 0.09314819831918818 * X**1+ 0.046950699273469274
"""


def error(file,n,l,method):
    """Prints the error using a given method
    :param file: a csv file with 2 columns, X and Y
    :param n: n-1 is the degree that we want to use for the regression
    :param l: real coefficient
    :param method: str, 'LSE' or 'Newton'
    """
    [A,B] = data(file,n)
    if method == "LSE":
        [x,R] = LSE(file,n,l)
    elif method == "Newton":
        [x,R,c] = Newton(file,n)
    else:
        raise ValueError("Please entre a valid method (LSE or Newton)")
    s = 0
    m = len(A)
    for i in range(m):
        s += (R[i][0]-B[i][0])**2
    print('the error is '+str(s))

"""
error(2,0)   => 16335.123164957964
error(3,0)   => 26.559959499333065
error(3,10000)    =>22649.738493024153
"""






#b. Newton's method

def Newton(file,n):
    [A,B]=data(file,n)
    X=[]
    for i in range(n):
        X.append([1]) #vector X0
    At=transpose(A)
    AtA=multiplication(At,A)
    C0=multiplication(AtA,X) #operations to find the gradient g=2AtAX
    C=scalar_multi(2,C0)
    D0=multiplication(At,B)
    D=scalar_multi(2,D0)
    grad=substraction(C,D)  
    H=scalar_multi(2,AtA) #H=Hessian matrix H=2AtA
    Hi=inverseLU(H) #We use the LU decomposition to find the inverse of H
    c=1
    F=multiplication(Hi,grad)
    X,Y=substraction(X,F),copy.deepcopy(X)  #X1=X0-F and we save X0 thanks to Y
    res=different(X,Y)
    while res==True and c<1000:
        F=multiplication(Hi,grad)
        X,Y=substraction(X,F),copy.deepcopy(X)
        res=different(X,Y)
        c+=1
    R=multiplication(A,X)
    return [X,R,c]
""" 
    [i,a]=inverse(AtA)
    R=multiplication(a,D0)
"""
    




def regression(file,n,l,method):
    """
    Prints the equation and the error given by the previous functions
    """
    [A,B]=data(file,n)
    if method=="LSE":
        [x,R]=LSE(file,n,l)
    elif method=="Newton":
        [x,R,c]=Newton(file,n)
    else :
        return "Please enter a valid method"
    equation(file,n,l,method)
    error(file,n,l,method)


""" I type regression("testfile.txt",n,0,"Newton") and I get : 
for n=2, l=0:
    4.432950310076807 * X**1+29.306404706056256
the error is 16335.123164957964
Same results as LSE
for n=3, l=0:
    the equation of the best ﬁtting line is :
3.0238533934865712 * X**2+ 4.906190263863799 * X**1+ -0.2314017560877204
he error is 26.559959499333065
Same results as LSE
for n=3, l=1000
the equation of the best ﬁtting line is :
3.0238533934865712 * X**2+ 4.906190263863799 * X**1+ -0.2314017560877204
the error is 26.559959499333065
l=lambda is not used in Newton method so we have the same results as previously
the results are better than the LSE method because the latter gave too much importance in minimizing the norm of X
"""
    




"""
The function to optimize is f=|AX-B|**2
f=XtAtAX-2XtAtb+btb
grad(f)=2AtAX-2AtB
WE want to minimize f so we are looking for the X verifying grad(f)=0
But grad F is linear, so it only takes one iteration to find the targeted X
This is why c=1
"""





def plot_regression(file, n, l, method):
    """
    Plots the results of the regression
    """
    [A,B]=data(file,n)
    if method == "LSE":
        [x,R] = LSE(file,n,l)
    elif method == "Newton":
        [x,R,c] = Newton(file,n)
    else :
        raise ValueError("Please enter a valid method")
    print(R)
    m = len(A)
    X = []
    Y = []
    Z = []
    for i in range(m):
        X.append(A[i][n-2])
        Y.append(B[i])
        Z.append(R[i])
    plt.scatter(X, Y, c = 'red')
    plt.plot(X, Z, c = 'yellow')



