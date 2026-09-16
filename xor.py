import numpy as np
import matplotlib.pyplot as plt

nodes = np.random.rand(2, 2)
weights = np.random.rand(2, 2 ,2)
grads = np.zeros((2,2,2))
bias = np.random.rand(3)
gradbias = np.zeros(3)

epoch = 500000
learningrate = 0.1

losses = np.zeros(epoch)

def xor(i1, i2):
    return i1 ^ i2

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])


def frontward(a, b):
    nodes[0,0] = a
    nodes[0,1] = b

    nodes[1,0] = sigmoid(nodes[0,0] * weights[0,0,0] + nodes[0,1] * weights[0,1,0] + bias[0])
    nodes[1,1] = sigmoid(nodes[0,0] * weights[0,0,1] + nodes[0,1] * weights[0,1,1] + bias[1])
    y = sigmoid(nodes[1,0] * weights[1,0,0] + nodes[1,1] * weights[1,1,0] + bias[2])
    return y

def backpropagation(ans, y):
    dy = y - ans
    delta_out = (y - ans) * y * (1 - y)
    delta_h0 = delta_out * nodes[1,0] * (1 - nodes[1,0]) * weights[1,0,0]
    delta_h1 = delta_out * nodes[1,1] * (1 - nodes[1,1]) * weights[1,1,0]
    grads[0,0,0] = nodes[0,0] * delta_h0
    grads[0,0,1] = nodes[0,0] * delta_h1
    grads[0,1,0] = nodes[0,1] * delta_h0
    grads[0,1,1] = nodes[0,1] * delta_h1
    grads[1,0,0] = nodes[1,0] * delta_out
    grads[1,1,0] = nodes[1,1] * delta_out

    gradbias[2] = delta_out
    gradbias[0] = delta_h0
    gradbias[1] = delta_h1


    weights[1,0,0] -= grads[1,0,0] * learningrate
    weights[1,1,0] -= grads[1,1,0] * learningrate

    weights[0,0,0] -= grads[0,0,0] * learningrate
    weights[0,1,0] -= grads[0,1,0] * learningrate
    weights[0,0,1] -= grads[0,0,1] * learningrate
    weights[0,1,1] -= grads[0,1,1] * learningrate

    bias[2] -= gradbias[2] * learningrate
    bias[1] -= gradbias[1] * learningrate
    bias[0] -= gradbias[0] * learningrate
        


for i in range(epoch):
    a, b, ans = data[i % 4]
    y = frontward(a, b)
    losses[i] = 0.5 * (y - ans)**2
    backpropagation(ans, y)


print(frontward(0,0))
print(frontward(0,1))
print(frontward(1,0))
print(frontward(1,1))

plt.plot(losses)
plt.grid()
plt.show()