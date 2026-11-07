import numpy as np

x = np.array([[1,2,3],[0.1,0.2,0.3]])
w = np.array([[0.2,0.41],[0.11,0.02],[0.001,0.12]])

b = 4

def forward_sum(x,w,b):
    z = np.dot(x,w) + b
    return z

def activation(z):
    return (1)/(1+np.exp(-z))


y = np.array([[1, 0], [0, 1]])

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


epochs = 1000
for i in range(epochs):
    z = forward_sum(x,w,b)
    A = activation(z)
    loss = mse_loss(A,y)
    dw,db = backward_prop(x,A,y)
    w,b = updateweight(w,b,dw,db,learning_rate=0.1)

    if i % 100 == 0:
        print("Loss(epoch: ",i,"): ",loss)