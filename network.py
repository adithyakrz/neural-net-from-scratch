import numpy as np

def forward_sum(x,w,b):
    z = np.dot(x,w) + b
    return z


def activation(z):
    return (1)/(1+np.exp(-z))


def mse_loss(A,y):
    res = (np.subtract(A,y))**2

    return np.mean(res)


def backward_prop(x,A,y):
    error = A-y
    product = error * A * (1-A)
    db = np.mean(product,axis=0)
    res = np.dot(x.T,product)
    dw = res/x.shape[0]
    return dw,db


def updateweight(w,b,dw,db,learning_rate):
    new_w =  w - (learning_rate*dw)
    new_b = b - (learning_rate*db)
    return new_w,new_b