import numpy as np

data = np.loadtxt("mnist_train.csv", delimiter=',', skiprows=1)
test_data = np.loadtxt("mnist_test.csv", delimiter=',', skiprows=1)

labels = data[:, 0]
pixels = data[:, 1:]
pixels = pixels / 255

test_labels = test_data[:, 0]
test_pixels = test_data[:, 1:]
test_pixels = test_pixels / 255

HID_SIZE = 100
OUT_SIZE = 10
BATCH_SIZE = 5000

epoch = 500
learningRate = 0.05 *(50 / BATCH_SIZE)
dataNum = 60000
testDataNum = 1000

hid = np.zeros((BATCH_SIZE, HID_SIZE))
out = np.zeros((BATCH_SIZE, 10))

W1 = np.random.randn(784, HID_SIZE) * 0.01
W2 = np.random.randn(HID_SIZE, 10) * 0.01

b1 = np.zeros(HID_SIZE)
b2 = np.zeros(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#softmax는 claude꺼 가져옴
def softmax(z):
    # z shape: (N, num_classes)
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # 오버플로우 방지
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(out, ans_label):
    eps = 1e-9
    correct_probs = out[np.arange(out.shape[0]), ans_label.astype(int)]  # 각 샘플의 정답 클래스 확률
    return -np.mean(np.log(correct_probs + eps))

def forward(pixel_input):
    global W1, W2, b1, b2, hid, out

    hid = sigmoid(np.matmul(pixel_input, W1) + b1)
    out = softmax(np.matmul(hid, W2) + b2)
    return out


def backprop(pixel_input, ans_label):
    global W1, W2, b1, b2, hid

    ans_label = ans_label.astype(int)
    ans_out = np.zeros((BATCH_SIZE, 10))
    ans_out[np.arange(BATCH_SIZE), ans_label] = 1 #Batch size 맞게 행렬 만들기 위해 fancy labeling?사용
    d_output = (ans_out - out) # (softmax + cross-entropy 미분이 그냥 이거임)
    grad_W2 = np.matmul(np.transpose(hid), d_output)
    grad_b2 = np.sum(d_output, axis=0)
    d_hid = np.multiply(np.matmul(d_output, np.transpose(W2)), (np.multiply(hid, (np.ones(HID_SIZE) - hid))))
    grad_W1 = np.matmul(np.transpose(pixel_input), d_hid)
    grad_b1 = np.sum(d_hid, axis=0)

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


#테스트 (일단은 디버깅을 해야하며 새로 불러오기 귀찮으니 학습시킨거 그대로 사용)
correct = 0
for i in range(0, testDataNum, BATCH_SIZE):
    batch_x = test_pixels[i:i+BATCH_SIZE]
    batch_y = test_labels[i:i+BATCH_SIZE]
    preds = np.argmax(forward(batch_x), axis=1)   # 배치 전체 예측, shape (BATCH_SIZE,)
    correct += np.sum(preds == batch_y)

print(correct)