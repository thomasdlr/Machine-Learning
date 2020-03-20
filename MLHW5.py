
#Part 1

#first, let's get the input data and regroup it into a vector X
data=[]
X=[]
with open("input.data", "r") as fic:
    data.append(fic.readlines())
data=data[0]
n=len(data)
for i in range(n):
    z=data[i].split('\n')
    x,y=z[0].split(' ')
    X.append([float(x),float(y)])
import numpy as np
X=np.array(X)

#noise epsilon:
b=5 #b**-1 is the variance of the noise

def noise(n):
    N=[]
    for i in range(n):
        eps=np.random.normal(0,b**(-1))
        N.append(eps)
    return np.array(N).T



from scipy.spatial.distance import pdist, squareform
b=5
a=1



def kernel(X1, X2, l=0.1, sigma_f=0.1,a=5.5089):
    squaredist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * (1+0.5 / (a*l**2) * squaredist)**(-a)

from numpy.linalg import inv

X_s=np.arange(-60,60,0.1).reshape(1200,1)
X_train=X[:,0].reshape(34,1)
Y_train=(X[:,1]+noise(34)).reshape(34,1)


def posterior_predictive(X_s, X_train, Y_train, l=1, sigma_f=1,a=1209.58141362):
    C= kernel(X_train, X_train, l, sigma_f,a) + (b**(-1)) * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f,a)
    K_ss = kernel(X_s, X_s, l, sigma_f,a) + b**(-1) * np.eye(len(X_s))
    K_inv = inv(C)
    
    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s


mu_s, cov_s=posterior_predictive(X_s,X_train,Y_train)

import matplotlib.pyplot as plt

Lx=[]
Ly=[]
Lvp=[]
Lvm=[]
for i in range(1200):
    Lx.append(i/10-60)
    Ly.append(mu_s[i])
    Lvp.append(mu_s[i]+3*cov_s[i][i])
    Lvm.append(mu_s[i]-3*cov_s[i][i])

plt.scatter(X_train,Y_train)
plt.plot(Lx,Ly,c="blue")

plt.plot(Lx,Lvp,c="red")
plt.plot(Lx,Lvm,c="green")
plt.fill_between(x=Lx,y1=np.array(Ly).reshape(1200,),y2=np.array(Lvp).reshape(1200,),color="yellow")
plt.fill_between(x=Lx,y1=np.array(Ly).reshape(1200,),y2=np.array(Lvm).reshape(1200,),color="yellow")



axes = plt.gca()
axes.set_xlim([-65,65])
axes.set_ylim([-10,10])
plt.show()

from scipy.optimize import minimize


def likelihood(parameters):
    a,sigma_f,l=parameters
    C= kernel(X_train, X_train, l, sigma_f,a)
    C_inv=np.linalg.inv(C)
    like=-0.5*np.log(np.linalg.det(C))-0.5*(X_train.T.dot(C_inv)).dot(X_train)
    return like


res= minimize(likelihood,(1,1,1),bounds=((0.1,None),(0.1,None),(0.1,None)))






#Part 2


import csv
import matplotlib.pyplot as plt
import numpy as np

X_train=[]
with open("X_train.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        pixels=row[0].split(',')
        image=[]
        for x in pixels:
            image.append(float(x))
        X_train.append(image)
X_train=np.array(X_train)

Y_train=[]
with open("Y_train.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        label=float(row[0])
        Y_train.append(label)
Y_train=np.array(Y_train)

Y_train-=1 #labels go from 1 to 5, we make them go from 0 to 4


X_test=[]
with open("X_test.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        pixels=row[0].split(',')
        image=[]
        for x in pixels:
            image.append(float(x))
        X_test.append(image)
X_test=np.array(X_test)

Y_test=[]
with open("Y_test.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        label=float(row[0])
        Y_test.append(label)
Y_test=np.array(Y_test)
Y_test-=1


from libsvm.svm import svm_problem,svm_parameter, gen_svm_nodearray
from libsvm.svmutil import *
x=[]
for row in X_train:
    x.append(row)
y=Y_train
from svm import *

prob = svm_problem(y,x)


xtest=[]
for row in X_test:
    xtest.append(row)
ytest=Y_test


param = svm_parameter('-t ')
m = svm_train(prob, param) # m is a ctype pointer to an svm_model
p_label, p_acc, p_val = svm_predict(y, x, m)
p_label_test, p_acc_test, p_val_test = svm_predict(ytest, xtest, m)

def f(z):
    cc,gamma=z
    param= svm_parameter('-t 2 -g '+str(gamma)+' -c '+str(cc))
    m = svm_train(prob, param)
    p_label_test, p_acc_test, p_val_test = svm_predict(ytest, xtest, m)
    return p_acc_test[0]

from scipy.optimize import minimize

minimize(f,(1,1/5000))

param1_list = [1+i*10 for i in range(20)]
param2_list = [1/5000+i*10/5000 for i in range(20)] # not necessarily the same number of values
results_size = (len(param1_list), len(param2_list))
results = np.zeros(results_size, dtype = np.float)

for param1_idx in range(len(param1_list)):
  for param2_idx in range(len(param2_list)):
    param1 = param1_list[param1_idx]
    param2 = param2_list[param2_idx]
    results[param1_idx, param2_idx] = f((param1, param2))

max_index = np.argmax(results)



def kernelmix(X1, X2,gamma,c):
    squaredist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

    return np.exp(-gamma*squaredist)+squaredist+c



def g(z):
    gamma,c=z
    K_train=kernelmix(X_train,X_train,gamma,c)
    m = svm_train(Y_train, [list(row) for row in K_train], '-c 4 -t 4')

    p_label_test, p_acc_test, p_val_test = svm_predict(ytest, xtest, m)
    return 100-p_acc_test[0]

