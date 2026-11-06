import numpy as np

x = np.array([[1,2,3],[0.1,0.2,0.3]])
w = np.array([[0.2,0.41],[0.11,0.02],[0.001,0.12]])

b = 4

def forward_sum(x,w,b):
    z = np.dot(x,w) + b
    return z

def activation(z):
    return (1)/(1+np.exp(-z))
z = forward_sum(x,w,b)
print(z)

a = activation(z)
print(a)