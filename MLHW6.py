import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import random
from scipy.spatial.distance import cdist

#initialize the parameters
k=2#there are 2 clusters
gamma_s=10#distance
gamma_c=10#color

def read_data(dataset="image1.png"):
  img=Image.open(dataset)
  L=[]
  dist_data=np.zeros((10000,2),dtype=int)
  for i in range(100):
      for j in range(100):
          l=img.getpixel((j,i))
          L.append(list(l))
          dist_data[i*100+j]=[i,j]
  return np.array(L,dtype=np.int),dist_data

data,dist_data=read_data()
plt.imshow(data.reshape(100,100,3))

def initialization(data, k=k):
    initialize_method = "a"
    means = np.random.rand(k,3)*255
    if initialize_method == "a":   #we initialize every point randomly
        classif_prec = np.random.randint(k, size=data.shape[0])
    elif initialize_method == "b":      #we initialize each column randomly
        classif_prec = []
        for i in range(data.shape[0]):
            if i % 2 == 1:
                classif_prec.append(0)
            else:
                classif_prec.append(1)
        classif_prec = np.asarray(classif_prec)
    return means, np.asarray(classif_prec),1,0#1 for iteration, 0 for loss

def draw( classification, iteration,k=2):
  title = "Kernel-K-Means Iteration-" + str(iteration)
  plt.clf()
  plt.suptitle(title)
  plt.imshow(classification.reshape((100,100)))
  plt.show()

#compute a rbf kernel given a parameter gamma and an input data
def rbf_kernel(data,gamma=10):
  dist=cdist(data,data)
  square_dist=dist*dist
  kernel_data=np.exp(-gamma*square_dist)
  return kernel_data


def new_kernel(data,dist_data,gamma_s=gamma_s,gamma_c=gamma_c):
    ks=rbf_kernel(dist_data,gamma_s)
    kc=rbf_kernel(data,gamma_c)
    return ks*kc

def second_term_of_calculate_distance(data, kernel_data, classification, data_number, cluster_number, k):
	result = 0
	number_in_cluster = 0
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			number_in_cluster += 1
	if number_in_cluster == 0:
		number_in_cluster = 1
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			result += kernel_data[data_number][i]
	return -2 * (result / number_in_cluster)

def third_term_of_calculate_distance(kernel_data, classification, k):
	temp = np.zeros(k, dtype=np.float32)
	temp1 = np.zeros(k, dtype=np.float32)
	for i in range(0, classification.shape[0]):
		temp[classification[i]] += 1
	for i in range(0, k):
		for p in range(0, kernel_data.shape[0]):
			for q in range(p + 1, kernel_data.shape[1]):
				if classification[p] == i and classification[q] == i:
					temp1[i] += kernel_data[p][q]
	for i in range(0, k):
		if temp[i] == 0:
			temp[i] = 1
		temp1[i] /= (temp[i] ** 2)
	return temp1

def classify(data, kernel_data, classification):
	temp_classification = np.zeros([data.shape[0]], dtype=np.int)
	third_term = third_term_of_calculate_distance(kernel_data, classification,k)
	for i in range(0, data.shape[0]):
		temp = np.zeros([k], dtype=np.float32) # temp size: k
		for j in range(k):
			temp[j] = second_term_of_calculate_distance(data, kernel_data, classification, i, j,k) + third_term[j]
		temp_classification[i] = np.argmin(temp)
	return temp_classification


def calculate_error(classification, previous_classification):
	error = 0
	for i in range(classification.shape[0]):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(classification.shape[0]):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def kernel_k_means(data,dist_data, kernel_data,k=k):
	means, previous_classification, iteration, prev_error = initialization(data, k)
	draw(previous_classification, iteration,k)
	classification = classify(data, kernel_data,means, previous_classification) 
	error = calculate_error(classification, previous_classification)
	while True:
		draw(classification, iteration,k)
		plt.imshow(classification.reshape((100,100)))
		iteration += 1
		previous_classification = classification
		classification = classify(data, kernel_data,means,  classification)
		error = calculate_error(classification, previous_classification)
		print(error)
		if error == prev_error:
			break
		prev_error = error
	means = update(data,dist_data, means, classification)
	draw( classification, iteration,k)
	print("Elapsed Time: {}".format(time.time() - start_time))
	print("Iterations: {}".format(str(iteration)))

if __name__ == "__main__":
    start_time = time.time()
    kernel_data = new_kernel(data, dist_data,gamma_s,gamma_c)
    print("The kernel has been computed.")
    kernel_k_means(data,dist_data, kernel_data)