from math import sqrt

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

#1. Generate n data points: , where  and x and y are independently sampled from N(mx1,vx1) and N(my1,vy1) respectively.
def Data1(n=50,mx1=1,vx1=2,my1=1,vy1=2):
    Dx=[]
    Dy=[]
    for i in range(n):
        x=unigaussian(mx1,vx1)
        y=unigaussian(my1,vy1)
        Dx.append(x)
        Dy.append(y)
    return [Dx,Dy]

#2. Generate n data points: , where  and x and y are independently sampled from N(mx1,vx1) and N(my1,vy1) respectively.
def Data2(n=50,mx2=10,vx2=2,my2=10,vy2=2):
    Dx=[]
    Dy=[]
    for i in range(n):
        x=unigaussian(mx2,vx2)
        y=unigaussian(my2,vy2)
        Dx.append(x)
        Dy.append(y)
    return [Dx,Dy]


import matplotlib.pyplot as plt

#This function will plot the two clusters
def groundtruth(n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=10,vx2=2,my2=10,vy2=2):
    [Dx1,Dy1]=Data1(n,mx1,vx1,my1,vy1)
    [Dx2,Dy2]=Data2(n,mx2,vx2,my2,vy2)
    plt.scatter(Dx1,Dy1,c="blue")
    plt.scatter(Dx2,Dy2,c="red")




#merges the two cluster in order to perform the classification
def merge(Dx1,Dy1,Dx2,Dy2):
    D=[]
    L=[]
    for i in range(len(Dx1)):
        x=Dx1[i]
        y=Dy1[i]
        label=0
        D.append([1,x,y])
        L.append(label)
    for i in range(len(Dx2)):
        x=Dx2[i]
        y=Dy2[i]
        label=1
        D.append([1,x,y])
        L.append(label)
    return [D,L]

import copy
import random

#shuffles the data
def shuffle(D,L): #D for data
    Dc=copy.deepcopy(D)
    Lc=copy.deepcopy(L)
    n=len(Dc)
    assert n==len(Lc)
    SD=[] #S for shuffled data
    SL=[]
    while n!=0:
        i=random.randint(0,n-1)
        SD.append(Dc[i])
        del(Dc[i])
        SL.append(Lc[i])
        del(Lc[i])
        n-=1
    return [np.array(SD),np.array(SL)]

import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

def proba(w,X):
    z=np.array(w[0]*np.array([1 for i in range(len(X))])+w[1]*np.array(X[:,1])+w[2]*np.array(X[:,2]))
    return sigmoid(z)

def costfunction(w, X, Y):
    y_pred = proba(w,X)
    return -1 * sum(Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred))
"""
def costfunction(w,X,L):
    Yprob=proba(w,X)
    for x in Yprob:
        if x==0:
            x=0.1
        if x==1:
            x=0.99
    costf=-1*np.add(L.dot(np.log(Yprob)),(1-L).dot(np.log(1-Yprob)))
    print(type(costf))
    return costf
"""
#Gradient Descent

def grad(w,X,L):
    Yprob=proba(w,X)
    n=len(X)
    g=[0]*3
    g[0]=1*np.array([1 for i in range(n)]).dot(np.subtract(Yprob,L))
    g[1]=1*X[:,1].T.dot(np.subtract(Yprob,L))
    g[2]=1*X[:,2].T.dot(np.subtract(Yprob,L))
    return g

"""or:
g[0]=sum(X[i,0]*(np.log(Yprob[i])-L[i]) for i in range(n))
g[1]=sum(X[i,1]*(np.log(Yprob[i])-L[i]) for i in range(n))
g[2]=sum((np.log(Yprob[i])-L[i]) for i in range(n))
    """



def gradientdescent(w,X,L,c):
    #print(w)
    costf=costfunction(w,X,L)
    if type(costf)==float:
        print(costf)
    #print("\n")
    g=grad(w,X,L)
    nextw0=w[0]-c*g[0]
    nextw1=w[1]-c*g[1]
    nextw2=w[2]-c*g[2]
    w=[nextw0,nextw1,nextw2]
    #if np.array_equal(np.array(w),np.array(nextw)):
    #    return nextw
    #w=nextw
    #print(g)
    return w

#arbitrarily initialize w:
w0=[-30,50,30]

nmax=100 #maximum number of iterations


def gradient(n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=10,vx2=2,my2=10,vy2=2,c=1):
    w=w0
    [Dx1,Dy1]=Data1(n,mx1,vx1,my1,vy1)
    [Dx2,Dy2]=Data2(n,mx2,vx2,my2,vy2)
    """plt.scatter(Dx1,Dy1,c="blue")
    plt.scatter(Dx2,Dy2,c="red")"""
    #boundary(w,n,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2,c)
    [Dx1,Dy1]=Data1(n,mx1,vx1,my1,vy1)
    [Dx2,Dy2]=Data2(n,mx2,vx2,my2,vy2)
    [X,L]=merge(Dx1,Dy1,Dx2,Dy2)
    [X,L]=shuffle(X,L)
    print(costfunction(w0,X,L))
    i=0
    while i<nmax:
        w=gradientdescent(w,X,L,c)
        i+=1
    
    #boundary(w,n,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2,c)
    [c0,c1]=classification(w,X,L)
    
    
    return [w,c0,c1]


def boundary(w,n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=10,vx2=2,my2=10,vy2=2,c=0.1):
    
    Lx=np.linspace(-2,14,1000)
    Ly=np.array([-w[1]/w[2]*Lx[i]-w[1]/w[0] for i in range(1000)])
    plt.plot(Lx,Ly)


def classification(w,X,L):
    c0,c1=0,0
    for i in  range(len(X)):
        x=X[i]
        if np.array(w).dot(x)<=0:
            plt.scatter(x[1],x[2],c="blue")
            if L[i]==0:       
                c0+=1
        elif np.array(x).dot(x)>0:
            plt.scatter(x[1],x[2],c="red")
            if L[i]==1:
                c1+=1
    plt.show()
    return [c0,c1]

#Newton's method:

def newton(n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=10,vx2=2,my2=10,vy2=2,c=1):#c still an input in case of H non invertible
    [Dx1,Dy1]=Data1(n,mx1,vx1,my1,vy1)
    [Dx2,Dy2]=Data2(n,mx2,vx2,my2,vy2)
    [X,L]=merge(Dx1,Dy1,Dx2,Dy2)
    [X,L]=shuffle(X,L)
    m=len(X)
    w=w0
    D=np.zeros((m,m))
    i=0
    cgrad=0
    while i<nmax:
        for j in range(m):
            e=1*np.array(w).dot(X[j])
            D[j][j]=(1-sigmoid(e))**2
        H=(X.T.dot(D)).dot(X) #Hessian matrix
        det=np.linalg.det(H)  #If it's not inversible, we need gradient descent
        if det==0:
            w=gradientdescent(w,X,L,c)
            cgrad+=1
        else: #otherwise, Wn+1=Wn-Hinv*grad
            Yprob=proba(w,X)
            grad=X.T.dot(np.subtract(Yprob,L))
            w=np.subtract(w,np.linalg.inv(H).dot(grad))
        i+=1
    [c0,c1]=classification(w,X,L)
    """for i in range(len(X)):
        if np.array(w).dot(X[i])<=0 and L[i]==0:
            c0+=1
        elif np.array(w).dot(X[i])>0 and L[i]==1:
            c1=+1"""
    print("cgrad="+str(cgrad)) #number of times when the gradient descent was used
    return [w,c0,c1]


import tableprint as tp

def result(method,c=1,n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=10,vx2=2,my2=10,vy2=2):
    if method=="gradient":
        [w,c0,c1]=gradient(n,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2,c)
        print("Gradient descent :\n")

    if method=="newton":
        [w,c0,c1]=newton(n,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2,c)
    print("w")
    print(w,end="\n\n")
        
    print("Confusion Matrix :")
    headers=[" ","Predict c1","Predict c2"]
    r1=["Is c1",c0,n-c0]
    r2=["Is c2",n-c1,c1]
    tp.table([r1,r2],headers)

    print("Sensitivity (Successfully predict cluster 1):"+str(c0/n))
    print("Specificity (Successfully predict cluster 2):"+str(c1/n))
    return w









#PART 2

import gzip
import copy
from math import log
f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

#no problem with big/low endian because I read byte by byte


import numpy as np
f.read(16) #We set the cursor at the beggining of the pixel bytes
buf = f.read(image_size*image_size*num_images) #We store the values of the pixels in a buffer
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)#We get the data from the buffer, make sure that the pixels are betwenn 0 and 255 and return them as floats
data = data.reshape(num_images, image_size, image_size, 1) #WE reorganize the data in order to have a matrix of size num_images*1 containing matrices of size image_size**2 storinf the values of the pixels
f.close()
   
#read the first labels
"""
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
for i in range(0,15):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
f.close()
"""

#let's import the test data

num_images_test = 10000
ftest = gzip.open('t10k-images-idx3-ubyte.gz','r')
ftest.read(16) #We set the cursor at the beggining of the pixel bytes
buft = ftest.read(image_size*image_size*num_images_test) #We store the values of the pixels in a buffer
datatest = np.frombuffer(buft, dtype=np.uint8).astype(np.float32)#We get the data from the buffer, make sure that the pixels are betwenn 0 and 255 and return them as floats
datatest = datatest.reshape(num_images_test, image_size, image_size, 1) #WE reorganize the data in order to have a matrix of size num_images_test*1 containing matrices of size image_size**2 storing the values of the pixels
ftest.close()


import matplotlib.pyplot as plt

def imagination(i):
    image = np.asarray(data[i]).squeeze() 
    for j in range(image_size):
        for k in range(image_size):
            if image[j][k]>=128:
                image[j][k]=1
            else:
                image[j][k]=0
    for j in range(image_size):
        for k in range(image_size-1):
            print(int(image[j][k]),end=' ')
        print(int(image[j][image_size-1]),end='\n')
    print('\n\n')



#function bins returns a vector X. X[i*image_size+j] is equal to 0 if image[i][j]<128, and it is equal to 1 otherwise
def bins(image):
    B=[]
    for i in range(image_size):
        for j in range(image_size):
            if image[i][j]<128:
                B.append(0)
            else:
                B.append(1)
    return np.array(B)


def initproba(image):
    l=[[0.5 for i in range(image_size)] for i in range(image_size)]
    return l

#matrix of size nbimage * 10 storing the probas that each image belons to a label
def problabel():
    PL=np.zeros((num_images,10))
    return PL
"""
def estimation(): #E step, actualization of the prob matrix, with the parameter p of each pixel
def maximization(): #M step
"""


""" Résumons ce que je fois faire: (voir p444 textbook)
1) initialiser les 28*28 pik et muk (pk)
2) formule pour les actualiser (E step)
3) calculer les gamma(znk) (wik) (M step) 
  
"""

import random

def normalization(M):
    (D,K)=M.shape
    for d in range(D):
        s=0
        for k in range(K):
            s+=M[d][k]
        for k in range(k):
            M[d][k]=M[d][k]/s
    return M


def initialization():
    #First, lets create a matrix of size num_image*(image_size**2)
    #each line i represents the bin vector of the ith image of the set
    X=[]
    for i in range(num_images):
        image = np.asarray(data[i]).squeeze() 
        B=bins(image)
        X.append(B)

    #let's create The list P of the Pik
    #Initially, we set every value to 1/10
    P=np.array([1/10 for i in range(10)])

    #Now, let's create a matrix M of size (num_images**2)*10 that stores the predictive mu
    M=np.array([[random.uniform(0.25,0.75) for i in range(10)] for i in range(image_size**2)])
    normalization(M)

    return [np.array(X),M,P]

    




#let's calculate the repsonsibility of component k given data point X[n]

#in order to do it, we need to know the prob of muk knowing data point x:
def likelihood(X,M): #return a table L of size num_image(s* 10 such that L[n][k]=p(xn|muk)
    M=np.add(M,10**(-10)*np.ones(M.shape))
    return np.exp(X.dot(np.log(M))+(1-X).dot(np.log(1-M)))



#en notation indicielle:
def update_responsibility_ind(X,P,M):
    L=likelihood(X,M)
    newG=np.zeros((num_images,10))
    for n in range(num_images):
        denominator=P.dot(L[n])
        #print(type(denominator))
        #assert type(denominator)==float
        for k in range(10):
            newG[n][k]=P[k]*L[n][k]/denominator
    #print(newG.shape)
    return newG



def effective_number(G):#returns a list N of size 10 such that N[k]=Nk
    N=G.T.dot(np.array([1 for i in range(num_images)])).T
    assert len(N)==10
    return N



#essayer avec des indices
def estimation(X,G,M):
    N=effective_number(G)
    #print(N)
    newM=copy.deepcopy(M)
   
    for k in range(10):
        newM[:,k]=1/N[k]*G[:,k].T.dot(X)
    assert newM.shape==M.shape
    

    newP=1/num_images*N
    return [newM,newP]





#[X,M,P]=initialization()
def EM(nb,X,M,P):
    for i in range(nb):
        G=update_responsibility_ind(X,P,M)
        [M,P]=estimation(X,G,M)
        #print(M,end="\n\n")
        #print(P,end="\n\n\n\n")
        if i%5==0 and i!=0:
            newrate=testEM(G,D)
            print(newrate)
            if i!=5:
                if abs(newrate-rate)<0.000001:
                    print(i)
                    break
            rate=newrate
    print(newrate)
    return G

#needed to use munkres
def cost(l,p,G):
    r=0
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8)
    for i in range(num_images):
        
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        if l==np.argmax(G[i]):
            if p==labels:
                r+=1
    f.close()
    return num_images-r #costfunction



from munkres import Munkres, print_matrix

def Munkres_matrix(G):
    K=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            K[i][j]=cost(i,j,G)
    return K



"""#To find the which class coresponds to chich digit
K=Munkres_matrix(G)
m=Munkres()
indexes = m.compute(matrix)
print(indexes)
"""
#It leads us to the followinf dictionnary    
D={0:7,1:4,2:3,3:6,4:5,5:8,6:9,7:1,8:2,9:0}


def testEM(G,D):
    r=0
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8)
    for i in range(num_images):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        l=np.argmax(G[i])
        if D[l]==labels[0]:
            r+=1
    f.close()
    return r/num_images #rate


def class_imagination(D):
    for i in range(10):
        print("class"+str(i))
        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        for j in range(num_images):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if D[labels[0]]==i:
                imagination(j)
                break
        f.close()
    for i in range(10):
        print("labelled class "+str(i))
        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        for j in range(num_images):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if labels[0]==i:
                imagination(j)
                break
        f.close()
        

def performance(lab,D,G):
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8)
    nblabel=0
    nbtrue=0
    nbfalse=0
    for i in range(num_images):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        if labels[0]==lab:
            nblabel+=1
            if D[np.argmax(G[i])]==labels[0]:
                nbtrue+=1
        else:
            if D[np.argmax(G[i])]==lab:
                nbfalse+=1
    f.close()
    return [nblabel,nbtrue,nbfalse]


import tableprint as tp

def confusion_matrix(D,G):
    for i in range(10):
        print("Confusion Matrix "+str(i)+":")
        headers=[" ","Predict nb"+str(i),"Not predict nb"+str(i)]
        [nblabel,nbtrue,nbfalse]=performance(i,D,G)
        r1=["Is nb"+str(i),nbtrue,nblabel-nbtrue]
        r2=["Is'nt nb"+str(i),nbfalse,num_images-nbfalse-nblabel]
        tp.table([r1,r2],headers)
        print("Sensitivity (Successfully predict number "+str(i)+"):"+str(nbtrue/nblabel))
        print("Specificity (Successfully predict not number "+str(i)+"):"+str((num_images-nbfalse-nblabel)/(num_images-nblabel)),end="\n\n\n")
