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

def forward_mlp(X, w1, b1, w2, b2):
  z1 = forward_sum(X, w1, b1)
  a1 = activation(z1)

  z2 = forward_sum(a1, w2, b2)
  a2 = activation(z2)

  return a1, a2


def backward_mlp(X, a1, a2, y, w2):
  delta2 = (a2 - y) * (a2 * (1 - a2))
  dw2 = np.dot(a1.T, delta2)
  db2 = np.sum(delta2, axis=0)

  delta1 = np.dot(delta2, w2.T) * (a1 * (1 - a1))
  dw1 = np.dot(X.T, delta1)
  db1 = np.sum(delta1, axis=0)

  return dw1, db1, dw2, db2