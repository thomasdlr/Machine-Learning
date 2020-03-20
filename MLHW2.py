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



#this function will return the 32 bin associated to an image
def classification(image):
    L=[0 for i in range(32)]
    for i in range(image_size):
        for j in range(image_size):
            c=int(image[i][j]//8)
            L[c]+=1
    return L


#we build a list such as LL[i] countains the list of classification lists having label=i
def classlabel():
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8)
    """c=0 #counts the number of times the label is lab"""
    LL=[[] for i in range(10)] 
    for i in range(num_images):
        image = np.asarray(data[i]).squeeze()
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        label=labels[0] #We get the label linked to this image
        L=classification(image)
        LL[label].append(L)
    f.close()
    return LL


LL=classlabel()


def posterior(classif):
    F=[[] for i in range(0)]#F will store the frequency of each bin for each label
    for i in range(10):
        n=len(LL[i])
        f=[] #f will store the frequecy with which the first bin has this value knowing that label=i
        for k in range(32):
            c=0
            for j in range(n):
                if LL[i][j][k]==classif[k]:
                    c+=1
                if c==0:
                    c=1
            f.append(c/n)
        F.append(f)
    P=[]
    for i in range(10):
        likelihood=1
        for x in F[i]:
            likelihood=likelihood*x
        prior=len(LL[i])/num_images
        P.append(log(likelihood*prior))
        s=0
    for i in range(10):
        s+=P[i]
    for i in range(10):
        P[i]=P[i]/s
    return P

def imagination(nb):
    for i in range(nb) :
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
        
def testdiscrete(nb):
    Lret=[]
    f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
    f.read(8)
    c=0
    for i in range(nb):
        image = np.asarray(datatest[i]).squeeze()
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        label=labels[0]
        cl=classification(image)
        L=posterior(cl)
        d=len(L)
        min=L[0]
        argmin=0
        for j in range(d):
            if L[j]<min:
                argmin=j
                min=L[j]
        print([argmin,label])
        Lret.append([argmin,label])
    f.close()
    for i in range(len(Lret)):
        if Lret[i][0]==Lret[i][1]:
            c=c+1
    return c/nb

    #testdiscrete(10000) >>30.14%, error rate=69.86%
        




#2


LLcontinuous=[[[[]for i in range(image_size)]for i in range(image_size)] for i in range(10)]
ftest = gzip.open('t10k-labels-idx1-ubyte.gz','r')
ftest.read(8)
for i in range(num_images_test):  
    image = np.asarray(datatest[i]).squeeze()
    buf = ftest.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    label=labels[0]
    for j in range(image_size):
        for k in range(image_size):
            LLcontinuous[label][j][k].append(image[j][k])
f.close()


LLestimate=[[[[]for i in range(image_size)]for i in range(image_size)] for i in range (10)]
for i in range(10):
    for j in range(image_size):
        for k in range(image_size):
            s=0
            N=len(LLcontinuous[i][j][k])
            for x in LLcontinuous[i][j][k]:
                s+=x
            mu=s/N
            ss=0
            for x in LLcontinuous[i][j][k]:
                ss+=(x-mu)**2
            v=ss/N
            if v==0: #get rid of the division by 0
                v=1
            LLestimate[i][j][k]=[N,mu,v]





from math import exp
from math import sqrt

def conti(nb):
    Lret=[]
    for i in range(0,nb):
        image = np.asarray(datatest[i]).squeeze()
        Lpost=[0 for i in range (10)]
        for j in range(10):
            loglike=1
            for k in range(image_size):
                for l in range(image_size):
                    N=LLestimate[j][k][l][0]
                    mu=LLestimate[j][k][l][1]
                    v=LLestimate[j][k][l][2]
                    loglike=loglike+log(1/sqrt(2*np.pi*v))-(image[k][l]-mu)**2/(2*v)
            prior=N/num_images
            logprior=log(prior)
            logpost=loglike+logprior
            Lpost[j]=logpost
        s=0
        for j in range(10):
            s=s+Lpost[j]
        for j in range (10):
            Lpost[j]=Lpost[j]/s
        Lret.append(Lpost)
    return Lret

def testconti(nb):
    L=conti(nb)
    fconti = gzip.open('t10k-labels-idx1-ubyte.gz','r')
    fconti.read(8)
    c=0
    for i in range(nb):
        buf = fconti.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        label=labels[0]    
        
        min=L[i][0]
        argmin=0
        for j in range(1,10):
            if L[i][j]<min:
                min=L[i][j]
                argmin=j
        if argmin==label:
            c=c+1
        print([argmin,label]) 
    f.close()
    return c/nb

#testconti(10000) >>71,36%, error rate=28.64%



        
##2 ONLINE LEARNING

from math import factorial

def comb(m,n):
    return factorial(n)/(factorial(m)*factorial(n-m))


#file : cointest.txt

def binomiallikelihood(fichier,a,b):
    f=open(fichier,"r")
    X=f.read()
    Y=X.split('\n')
    m=len(Y)
    for i in range(m):
        k=0
        n=len(Y[i])
        for j in range(n):
            if int(Y[i][j])==1:
                k=k+1
            p=k/n
        likelihood=comb(k,n)*p**k*(1-p)**(n-k)
        print(Y[i])
        print("Likelihood:"+str(likelihood))
        print("Beta prior:  a ="+str(a)+" b ="+str(b)) 
        a=a+k
        b=b+n-k
        print("Beta posterior:  a="+ str(a)+" b="+str(b))
        print("\n")
    f.close()
    