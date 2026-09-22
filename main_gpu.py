import numpy as np
import cupy as cp

data = np.loadtxt("mnist_train.csv", delimiter=',', dtype=np.float32, skiprows=1)
test_data = np.loadtxt("mnist_test.csv", delimiter=',', dtype=np.float32, skiprows=1)
labels = cp.asarray(data[:, 0])
pixels = cp.asarray(data[:, 1:])
pixels = pixels / 255

test_labels = cp.asarray(test_data[:, 0])
test_pixels = cp.asarray(test_data[:, 1:])
test_pixels = test_pixels / 255

HID_SIZE = 100
OUT_SIZE = 10
BATCH_SIZE = 5000

epoch = 500
learningRate = 0.05 *(50 / BATCH_SIZE)
dataNum = 60000

W1 = cp.random.randn(784, HID_SIZE,dtype=np.float32) * 0.01
W2 = cp.random.randn(HID_SIZE, 10,dtype=np.float32) * 0.01

b1 = cp.zeros(HID_SIZE,dtype=np.float32)
b2 = cp.zeros(10,dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

#softmax는 claude꺼 가져옴
def softmax(z):
    # z shape: (N, num_classes)
    z_shifted = z - cp.max(z, axis=1, keepdims=True)  # 오버플로우 방지
    exp_z = cp.exp(z_shifted)
    return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(out, ans_label):
    eps = 1e-9
    correct_probs = out[cp.arange(out.shape[0]), ans_label.astype(int)]  # 각 샘플의 정답 클래스 확률
    return -cp.mean(cp.log(correct_probs + eps))

def forward(pixel_input):
    global W1, W2, b1, b2, hid, out

    hid = sigmoid(cp.matmul(pixel_input, W1) + b1)
    out = softmax(cp.matmul(hid, W2) + b2)
    return out


def backprop(pixel_input, ans_label):
    global W1, W2, b1, b2, hid

    ans_label = ans_label.astype(int)
    ans_out = cp.zeros((BATCH_SIZE, 10),dtype=np.float32)
    ans_out[cp.arange(BATCH_SIZE), ans_label] = 1 #Batch size 맞게 행렬 만들기 위해 fancy labeling?사용
    d_output = (ans_out - out) # (softmax + cross-entropy 미분이 그냥 이거임)
    grad_W2 = cp.matmul(cp.transpose(hid), d_output)
    grad_b2 = cp.sum(d_output, axis=0)
    d_hid = cp.multiply(cp.matmul(d_output, cp.transpose(W2)), (cp.multiply(hid, (cp.ones(HID_SIZE,dtype=np.float32) - hid))))
    grad_W1 = cp.matmul(cp.transpose(pixel_input), d_hid)
    grad_b1 = cp.sum(d_hid, axis=0)

    #update
    W1 += grad_W1 * learningRate
    W2 += grad_W2 * learningRate
    b1 += grad_b1 * learningRate
    b2 += grad_b2 * learningRate


#학습

for i in range(epoch):
    for j in range(0, dataNum, BATCH_SIZE):
        forward(pixels[j:j+BATCH_SIZE])
        loss = cross_entropy_loss(out, labels[j:j+BATCH_SIZE])
        backprop(pixels[j:j+BATCH_SIZE], labels[j:j+BATCH_SIZE])
    print(i, loss)


preds = cp.argmax(forward(test_pixels), axis=1)
correct = cp.sum(preds == test_labels)
print(correct / 10000)