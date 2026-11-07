import numpy as np
from network import forward_sum, activation, mse_loss, backward_prop, updateweight

x = np.array([[1,2,3],[0.1,0.2,0.3]])
w = np.array([[0.2,0.41],[0.11,0.02],[0.001,0.12]])
y = np.array([[1, 0], [0, 1]])

b = 4

epochs = 1000
for i in range(epochs):
    z = forward_sum(x,w,b)
    A = activation(z)
    loss = mse_loss(A,y)
    dw,db = backward_prop(x,A,y)
    w,b = updateweight(w,b,dw,db,learning_rate=0.1)

    if i % 100 == 0:
        print("Loss(epoch: ",i,"): ",loss)