import numpy as np
from math import exp, sqrt
import copy
import matplotlib.pyplot as plt
import random

## 1. Random data generator
##  a. Univariate Gaussian data generator

def unigaussian(m,s):
    L=[]
    for i in range(12):
        L.append(np.random.uniform(low=0.0, high=1.0, size=None))
    z=0
    for i in range(12):
        z+=L[i]
    z-=6
    x=m+sqrt(s)*z
    return x

"""
import matplotlib.pyplot as plt
X=[]
Y=[]
for i in range(50):
    x=unigaussian(0,1)
    X.append(x)
    Y.append(0)
plt.scatter(X,Y)
"""

##  b.  Polynomial basis linear model data generator
def vector(x,n):
    v=np.array([[x**i] for i in range(n)])
    return v

def poly(a,w,x):
    n=w.shape[0]
    e=unigaussian(0,1/a)
    X=vector(x,n)
    y=np.dot(w,X)
    y+=e
    return float(y)

np.transpose(np.array([[0],[0]]))



## 2. Sequential estimator

def variance(X,mu):
    n=len(X)
    v=0
    for i in range(n):
        v=v+(X[i]-mu)**2
    v=v/n
    return v

nmax=10000
precision=0.05
def sequential(m,s):   
    x=unigaussian(m,s)
    mu=x
    X=[]
    v=0
    n=1
    while (abs(mu-m)>precision or abs(v-s)>precision) and n<nmax:
        x=unigaussian(m,s)
        X.append(x)
        mu=(n*mu+x)/(n+1)
        v=variance(X,mu)
        print("Add data point:"+str(x))
        print("Mean = "+str(mu))
        print("Variance = "+str(v)+"\n" )
        n=n+1
    return [mu,v,n]

#3

def posteriorvariance(a,X,L):
    Xt=X.T
    M1=a*Xt.dot(X)
    M2=np.add(M1,L)
    return M2

def posteriormean(a,C,X,Y): 
    Ci=np.linalg.inv(C)
    Xt=X.T
    M1=Ci.dot(Xt)
    M2=M1.dot(Y)
    return a*M2

def predictivevariance(a,X,L):
    Xt=X.T
    M1=Xt.dot(np.linalg.inv(L))
    M2=M1.dot(X)
    lambdai=1/a+float(M2)
    return lambdai

def predictivemean(Mu,x):
    Mut=Mu.T
    M=Mut.dot(x)
    return float(M)

def bayesianplot(a,Lx,YY,Mu,L):
    n=len(Mu)
    plt.scatter(Lx,YY)
    Xlist=np.linspace(-2,2,1000)
    Y=[]
    Yvp=[]
    Yvm=[]
    for i in range(1000):
        xx=vector(Xlist[i],n)
        y=predictivemean(Mu,xx)
        Y.append(y)
        yvp=y+predictivevariance(a,xx,L)
        Yvp.append(yvp)
        yvm=y-predictivevariance(a,xx,L)
        Yvm.append(yvm)
    plt.plot(Xlist,Y,c="blue",)
    plt.plot(Xlist,Yvp,c="red")
    plt.plot(Xlist,Yvm,c="red")
    plt.xlim(-2,2)
    plt.show()

def converge(V,precision): #V is a column array vector
    n=len(V)
    res=True
    for i in range(n):
        if V[i]>precision:
            res=False
    return res


def bayesianreg():
    a=3
    b=1
    n=3
    B=np.zeros((n,n))
    for i in range(n):
        B[i][i]=b
    w=np.array([1,2,3,4])
    Lx=[]
    XX=[]
    YY=[]
    Ym=[]
    Yvp=[]
    Yvm=[]
    L=copy.deepcopy(B)
    Mu=np.array([0 for i in range (n)])
    c=0
    cv=False
    while cv==False and c<15000:
        c+=1
        x=random.uniform(-1,1)
        Lx.append(x)
        y=poly(a,w,x)
        print("add data point"+str((x,y)))
        XX.append([x**i for i in range(n)]) #XX is the design matrix
        X=np.array(XX)
        YY.append(y)
        Yt=np.array(YY) #YY is a row vector, Y is a column array vector
        Y=Yt.T
        Prec=posteriorvariance(a,X,B)
        M=posteriormean(a,Prec,X,Y)
        print("posterior mean : "+str(M))
        xx=vector(x,n)
        L=copy.deepcopy(Prec)
        print("posterior variance : "+str(L)+str("\n"))
        Mu=copy.deepcopy(M)
        cv=converge(abs(w-Mu),0.1)
        lambdai=predictivevariance(a,xx,L)
        meanpred=predictivemean(Mu,xx)
        #print("predictive : "+str(meanpred)+str(",")+str(lambdai)+str("\n"))
        Ym.append(meanpred)
        Yvp.append(meanpred+lambdai)
        Yvm.append(meanpred-lambdai)
        if c==10 or c==50:
            bayesianplot(a,Lx,YY,Mu,L)
    bayesianplot(a,Lx,YY,Mu,L)
    print(c)
    return (Mu,L)
"""
(Mu,L)=bayesianreg(500)
n=4
a=1
x=np.linspace(-2,2,1000)
xx=vector(x,n)
Y=[]
Yvp=[]
Yvm=[]
for i in range(1000):
    y=predictivemean(Mu,xx)
    Y.append(y)
    yvp=y+predictivevariance(a,xx,L)
    Yvp.append(yvp)
    yvm=y-predictivevariance(a,xx,L)
    Yvm.append(Yvm)
    plt.plot(Lx,Ym,c="blue")
    plt.plot(Lx,Yvp,c="red")
    plt.plot(Lx,Yvm,c="red")
"""
