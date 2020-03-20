
import copy
import matplotlib.pyplot as plt
import numpy as np

# a. For LSE
# 1

a=1

def transpose(A): 
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
    M=[]
    for i in range(n):
        L=[]
        for j in range(m):
            L.append(0)
        M.append(L)
    return M



def multiplication(A,B): 
    n=len(A)
    l=len(B)
    p=len(A[0])
    m=len(B[0])
    if p!=l:
        return 'error'
    C=zeros(n,m)
    for i in range (n):
        for j in range(m):
            for k in range(p):
                C[i][j]+=A[i][k]*B[k][j]
    return C


def to_be_inversed(A,l): #return AtA+lI
    B=transpose(A)
    n=len(A[0])
    C=multiplication(B,A)
    for i in range(n):
        C[i][i]+=l
    return C

def substraction(A,B): 
    m=len(A)
    n=len(A[0])
    C=zeros(m,n)
    for i in range(m):
        for j in range(n):
            C[i][j]=A[i][j]-B[i][j]
    return C


def LU(M): #M is a square matrix of size n
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




 

def inverse_trilow(A): #Gauss method to inverse a lower triangular matrix
    n=len(A)
    T=copy.deepcopy(A)
    I=zeros(n,n)
    for i in range(n):
        I[i][i]=1
    for k in range(n-1):
        pivot=T[k][k]
        for i in range(k+1,n):
            for j in range(n):
                T[i][j],I[i][j]=T[i][j]-T[i][k]/pivot*T[k][j],I[i][j]-T[i][k]/pivot*I[k][j]
    for k in range(n):
        if T[k][k]!=1:
            c=T[k][k]
            for j in range(n):
                I[k][j]=I[k][j]/c
                T[k][j]=T[k][j]/c
    return I  


def inverseLU(A): #use of LU decomposition to inverse a matrix A
    #I will use it in the Newton method
    [L,U]=LU(A)
    Ut=transpose(U)
    Li=inverse_trilow(L)
    Uti=inverse_trilow(Ut)
    Ui=transpose(Uti)
    Inv=multiplication(Ui,Li)
    return Inv
          

def data(name,n):
    A=[]
    B=[]
    fichier=open(name,"r")
    X=fichier.read()
    Y=X.split(' ')
    m=len(Y)
    for i in range(m):
        l=[]
        Z=Y[i].split(',')
        a=float(Z[0])
        for k in reversed(range(n)):
            l.append(a**k)
        A.append(l)
        B.append([float(Z[1])])
    return [A,B]   


def LSE(name,n,l): #this function finds the X verifying (AtA)**(-1)*X=AtB
    [A,B]=data(name,n)
    C=to_be_inversed(A,l)
    [L,U]=LU(C)
    m=len(C)
    At=transpose(A)
    D=multiplication(At,B)
    p=len(D[0])
    Y=zeros(m,p) # We define Y=UX and solve LY=D
    for i in range(m):
        for j in range(p):
            Y[i][j]=D[i][j]
            if i!=0:
                for k in range(i):
                    Y[i][j]-=L[i][k]*Y[k][j]
    X=zeros(m,p) #We find X from Y=UX
    for i in reversed(range(m)):
        for j in range(p):
            X[i][j]=1/U[i][i]*Y[i][j]
            if i!=m-1:
                for k in range(i+1,m):
                    X[i][j]-=1/U[i][i]*U[i][k]*X[k][j]
    R=multiplication(A,X)
    return [X,R]
    
    
                    



#a.2

def equation(name,n,l,methode):
    if methode=="LSE":
        [X,R]=LSE(name,n,l)
    elif methode=="Newton":
        [X,R,c]=Newton(name,n)
    else :
        return "Please enter a valid method"
    print('the equation of the best ﬁtting line is :')
    for i in range(n-1):
        print(str(X[i][0])+ ' * X**'+str(n-i-1)+'+')
    print(str(X[n-1][0]))



"""
equation(2,0)   => 4.432950310076807 * X**1+ 29.306404706056256
equation(3,0)   => 3.0238533934865712 * X**2+ 4.906190263863799 * X**1  -0.23140175608771987
equation(3,10000)   => 0.8345332827002858 * X**2+ 0.09314819831918818 * X**1+ 0.046950699273469274
"""


def error(name,n,l,methode):
    [A,B]=data(name,n)
    if methode=="LSE":
        [x,R]=LSE(name,n,l)
    elif methode=="Newton":
        [x,R,c]=Newton(name,n)
    else :
        return "Please enter a valid method"
    s=0
    m=len(A)
    for i in range(m):
        s+=(R[i][0]-B[i][0])**2
    print('the error is '+str(s))

"""
error(2,0)   => 16335.123164957964
error(3,0)   => 26.559959499333065
error(3,10000)    =>22649.738493024153
"""



def different(A,B):
    m=len(A)
    n=len(A[0])
    res=False
    for i in range(m):
        for j in range(n):
            if A[i][j]==B[i][j]:
                res=True
    return res

def scalar_multi(l,M):
    m=len(M)
    n=len(M[0])
    for i in range(m):
        for j in range(n):
            M[i][j]=l*M[i][j]
    return M


#b. Newton's method

def Newton(name,n):
    [A,B]=data(name,n)
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
    




def regression(name,n,l,methode): #this function prints the equation and the error given by the previous functions
    [A,B]=data(name,n)
    if methode=="LSE":
        [x,R]=LSE(name,n,l)
    elif methode=="Newton":
        [x,R,c]=Newton(name,n)
    else :
        return "Please enter a valid method"
    equation(name,n,l,methode)
    error(name,n,l,methode)

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





def trace(name,n,l,methode):
    [A,B]=data(name,n)
    if methode=="LSE":
        [x,R]=LSE(name,n,l)
    elif methode=="Newton":
        [x,R,c]=Newton(name,n)
    else :
        return "Please enter a valid method"
    print(R)
    m=len(A)
    X=[]
    Y=[]
    Z=[]
    for i in range(m):
        X.append(A[i][n-2])
        Y.append(B[i])
        Z.append(R[i])
    plt.scatter(X, Y, c = 'red')
    plt.plot(X, Z, c = 'yellow')









   
   



    
            

