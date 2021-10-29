import pandas as pd
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

LR = 1   

I_dim = 3
H_dim = 4

epoch_count = 1

weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

preActivation_H = np.zeroes(H_dim)
postActivation_H = np.zeroes(H_dim)

training_data = pd.read_excel('MLP_Tdata.xlsx')
target_output = training_data.output
training_data = training_data.drop(['output'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:,0])

validation_data = pd.read_excel('MLP_Vdata.xlsx')
validation_output = validation_data.output
validation_data = validation_data.drop(['output'], axis=1)
validation_data = np.asarray(validation_data)
validation_count = len(validation_data[:,0])


def train():
    for epoch in range(epoch_count):
        for sample in range(training_count):
            for node in range(H_dim):
                preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
                postActivation_H[node] = sigmoid(preActivation_H[node])
                
            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = sigmoid(preActivation_O)
            
            FE = postActivation_O - target_output[sample]
            
            for H_node in range(H_dim):
                S_error = FE * sigmoid(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[H_node]
                        
                for I_node in range(I_dim):
                    input_value = training_data[sample, I_node]
                    gradient_ItoH = S_error * weights_HtoO[H_node] * sigmoid(preActivation_H[H_node]) * input_value
                    
                    weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
                    
                weights_HtoO[H_node] -= LR * gradient_HtoO