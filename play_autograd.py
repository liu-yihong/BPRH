import autograd.numpy as np

A = np.array([[3, 4, 5, 2],
                   [4, 4, 3, 3],
                   [5, 5, 4, 4]], dtype=np.float32).T

shape = A.shape
rank = 2

def cost(W, H):
    pred = np.dot(W, H)
    return np.sqrt(((pred - A).flatten() ** 2).mean(axis=None))

from autograd import grad,

grad_cost= multigrad(cost, argnums=[0,1])

H =  np.random.randn(rank, shape[1])
W =  np.random.randn(shape[0], rank)

learning_rate=0.01

for i in range(5000):
    if i%50==0:
        print "*"*20
        print i
        print "*"*20
        print cost(W, H)
    del_W, del_H = grad_cost(W, H)
    W = W-del_W*learning_rate
    H = H-del_H*learning_rate

pred = np.dot(W, H)