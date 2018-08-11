import numpy as np
import csv

	

def init_network(*layer):
	result = []
	for i in range(len(layer)):
		result = result + [np.random.rand(layer[i])]		
	return result
	
	

def load_data(file, batchsize):
	with open(file, 'r') as f:
		headers = f.readline().split(',')		
		t = np.asarray([np.asarray(row.split(','), dtype=np.int32) for row in f])
		labels = t[:,0]
		data = t[:,1:]
				
	return data.reshape(len(data) // batchsize, batchsize, len(headers) - 1), labels.reshape(len(labels) // batchsize, batchsize)

def forward_prop(network, data):
	for layer in range(len(network)-1):		
		data = [np.dot(network[layer], data) for i in network[layer+1]]
	
	return [np.dot(network[-1], data) for i in range(len(network[-1]))]
		
		
		
network = init_network(784,8,9,10)
data, labels = load_data("Data/train_small.csv", 2)
#print(np.shape(labels))
#print(labels)
#print(np.dot(np.asarray(network[0]),data[0][0]))
print(forward_prop(network, data[0][0]))