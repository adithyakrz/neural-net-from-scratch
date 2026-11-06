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

A = activation(z)
print(A)

y = np.array([[1, 0], [0, 1]])

def mse_loss(A,y):
    res = (np.subtract(A,y))**2

    return np.mean(res)

loss = mse_loss(A,y)
print(loss)


def backward_prop(x,A,y):
    error = A-y
    product = error * A * (1-A)
    res = np.dot(x.T,product)
    dw = res/x.shape[0]
    return dw

dw = backward_prop(x,A,y)
print(dw)

learning_rate = 0.01

def updateweight(w,dw,learning_rate):
    return w - (learning_rate*dw)

w = updateweight(w,dw,learning_rate)
print(w)