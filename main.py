import pandas as pd
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


