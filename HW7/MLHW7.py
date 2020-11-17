from PIL import Image
import matplotlib.pyplot as plt


import glob
import numpy as np


def get_data():
    X,Y=np.zeros((135,2688)),[]
    i=0
    files = sorted(glob.glob('Yale_Face_Database/Training_resized/*'))
    for file in files:
      #print(file)
      Y.append(file.split(".")[0].split("/")[1])
      with Image.open(file) as im:
        data = list(im.getdata())
        X[i]=data
      i+=1
    return X,np.array(Y)


def get_test_data():
    X,Y=np.zeros((30,2688)),[]
    i=0
    files = sorted(glob.glob('Yale_Face_Database/Testing_resized/*'))
    for file in files:
      #print(file)
      Y.append(file.split(".")[0].split("/")[1])
      with Image.open(file) as im:
        data = list(im.getdata())
        X[i]=data
      i+=1
    return X,np.array(Y)


import copy
    
def eigenvec(M):
    eigen_values,eigen_vectors=np.linalg.eig(M)
    order=eigen_values.argsort()[::-1] #returns the index of the eigenvalues ordered from the largest to the smallest
    sorted_eigen_vectors=copy.deepcopy(eigen_vectors)
    for i in range(len(order)):
        sorted_eigen_vectors[:,order[i]]=eigen_vectors[:,i]
    return sorted_eigen_vectors[:,:2]#we return the two eigenvectors linked to the largest eigenvalues

def PCA():
	x, label = get_data() 
	covariance = np.cov(x.transpose()) 
	feature_vectors = eigenvec(covariance) 
	return feature_vectors


#****************LDA******************

def compute_mean(x, label):
	mean = np.zeros((15, x.shape[1]))
	for i in range(0, x.shape[0]):
		for j in range(0, x.shape[1]):
			mean[int(i / 9)][j] += x[i][j]
	for i in range(0, 15):
		for j in range(0, x.shape[1]):
			mean[i][j] /= 9
	mean_tot = np.mean(x, axis=0)
	return mean, mean_tot

def compute_within_class(x, mean):
	scat = np.zeros((x.shape[1], x.shape[1]))
	for i in range(0, x.shape[0]):
		centered_x = np.subtract(x[i], mean[i//9]).reshape(x.shape[1], 1)
		scat += np.matmul(centered_x, centered_x.transpose())
	return scat

def compute_between_class(mean, mean_tot):
	scat = np.zeros((x.shape[1], x.shape[1]))
	for i in range(0, 15):
		centered_mean = np.subtract(mean[i], mean_tot).reshape(x.shape[1], 1)
		scat += np.matmul(centered_mean, centered_mean.transpose())
	scat *= 9
	return scat


def draw(data, label):
  plt.title('LDA')
  for i in range(0, data.shape[0]):
    plt.scatter(data[i][0], data[i][1], s=4, )
  plt.show()

def LDA():
  x, label = get_data() # x size: 5000 * 784, label size: 5000 * 1
  mean, overall_mean = compute_mean(x, label) # mean size: 5 * 784
  within_class = compute_within_class(x, mean)
  between_class = compute_between_class(mean, overall_mean)
  M=np.linalg.pinv(within_class).dot(between_class)
  feature_vectors = eigenvec(M)
  lower_dimension_data = x.dot(feature_vectors)#np.matmul(x, feature_vectors)
  draw(lower_dimension_data, label)
  return feature_vectors

#**********************Kernel PCA*****************




#*********************Kernel LDA*******************








#***********************eigen/fisher faces*****************

def project(x,method):
  if method=="PCA":
    feature_vectors=PCA()
  elif method=="LDA":
    feature_vectors=LDA()
  projected_x=copy.deepcopy(x)
  for i in range(projected_x.shape[0]):
    projected_x[i,:]=np.add(x[i,:].dot(feature_vectors[:,0])*feature_vectors[:,0],x[i,:].dot(feature_vectors[:,1])*feature_vectors[:,1])
  return projected_x

def show(projected_x,i):
    image =projected_x[i].reshape((56,48))
    plt.figure()
    plt.imshow(image)


def faces(x,method):
  projected_x=project(x,method)
  for i in range(25):
    show(projected_x,i)

#********************face recognition****************************



def k_nearest_neighbors(array,K):#projected_x is the training data, xy the couple to classify, K the number of clusters
  l=[]
  for i in range(len(array)):
    lab=i//9
    dist=array[i]
    if l==[]:
      l.append([lab,dist])
    else:
      c=0
      while dist>l[c][1]and c<len(l)-1:
        c+=1
      l=l[:c]+[[lab,dist]]+l[c:]
  print(l)
  lk=np.zeros((15,))
  for k in range(K):
    lk[l[k][0]]+=1
  return np.argmax(lk)

def k_nearest_neighbors_kernel(array,K):#projected_x is the training data, xy the couple to classify, K the number of clusters
  l=[]
  for i in range(len(array)):
    lab=i//9
    dist=array[i]
    if l==[]:
      l.append([lab,dist])
    else:
      c=0
      while dist<l[c][1]and c<len(l)-1:
        c+=1
      l=l[:c]+[[lab,dist]]+l[c:]
  #print(l)
  lk=np.zeros((15,))
  for k in range(K):
    lk[l[k][0]]+=1
  return np.argmax(lk)

def classif(method,K=9):
  x,label=get_data()
  D=np.zeros((30,135))#stores the dist bw two points 
  x_test,label_test=get_test_data()
  l_label=np.zeros((x_test.shape[0],))
  if method=="PCA":
    feature_vectors=PCA()
  elif method=="LDA":
    feature_vectors=LDA()
  for i in range(x_test.shape[0]):
    for j in range(x.shape[0]):
      xy_test=x_test[i].dot(feature_vectors)
      xy=x[j].dot(feature_vectors)
      D[i,j]=(xy[0]-xy_test[0])**2+(xy[1]-xy_test[1])**2
  print(D)
  llabel=[]
  for row in D:
    llabel.append(k_nearest_neighbors(row,K))
    rate=0
  for i in range(len(llabel)):
    if llabel[i]==i//2:
      rate+=1
  return 100*rate/30

def best_K(method):
  l=np.zeros((10,))
  for i in range(5,15):
    l[i-5]=classif(method,i)
  return l


def classif_kernel(method,kernel_type,K=9,gamma=0.1):
  x,label=get_data()
  D=np.zeros((30,135))#stores the dist bw two points 
  x_test,label_test=get_test_data()
  l_label=np.zeros((x_test.shape[0],))
  if method=="PCA":
    feature_vectors=PCA()
  elif method=="LDA":
    feature_vectors=LDA()
  for i in range(x_test.shape[0]):
    for j in range(x.shape[0]):
      xy_test=x_test[i].dot(feature_vectors)
      xy=x[j].dot(feature_vectors)
      if kernel_type=="RBF":
        square_dist=(xy[0]-xy_test[0])**2+(xy[1]-xy_test[1])**2
        D[i,j]=np.exp(-square_dist*gamma)
      if kernel_type=="linear":
        temp=xy.dot(xy_test)
        D[i,j]=temp
  print(D)
  llabel=[]
  for row in D:
    llabel.append(k_nearest_neighbors_kernel(row,K))
    rate=0
  for i in range(len(llabel)):
    if llabel[i]==i//2:
      rate+=1
  return 100*rate/30

def best_K_kernel(method,kernel_type):
  l=np.zeros((10,))
  gamma=1e-4
  for i in range(5,15):
    l[i-5]=classif_kernel(method,kernel_type,K=i,gamma=gamma)
  return l

def best_gamma():
  l=[1e-4,1e-3,1e-2,10,100]
  lres=[]
  for gamma in l:
    lres.append(classif_kernel("PCA","quadratic",K=8,gamma=gamma))
  return np.argmax(lres)


"""
#**************LDA*************************
projected_x=copy.deepcopy(x)
for i in range(projected_x.shape[0]):
  projected_x[i,:]=np.add(x[i,:].dot(feature_vectors[:,0])*feature_vectors[:,0],x[i,:].dot(feature_vectors[:,1])*feature_vectors[:,1])
def show(projected_x,i):
    image =projected_x[i].reshape((56,48))
    plt.figure()
    plt.imshow(image)
for i in range(25):
    show(projected_x,i)
  
#********************face recognition****************************
def compare(projected_x,i,xy):#Compare the similarity of X[i]and x[j]
  diff=0
  for k in range(len(xy)):
    diff+=(projected_x[i,k]-xy[k])**2
    return diff
def k_nearest_neighbors(projected_x,xy,K):#projected_x is the training data, xy the couple to classify, K the number of clusters
  l=[]
  for im in range(projected_x.shape[0]):
    diff=compare(projected_x,im,i)
    if l==[]:
      l.append([im,diff])
    else:
      c=0
      while diff>l[c][1]and c<len(l)-1:
        c+=1
      l=l[:c]+[[im,diff]]+l[c:]
  lk=np.zeros((15,))
  for k in range(K):
    lk[l[k][0]//9]+=1
  return np.argmax(lk)
"""
