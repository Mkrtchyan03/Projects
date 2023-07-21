import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0

    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    def sigmoid_der(self, data):
        return self.sigmoid(data)*(1-self.sigmoid(data))

    def ReLU(self, data):
        return np.maximum(0, data)

    def der_relu(self, data):
        return np.where(data > 0, 1, 0)


    def adam(self, grd_w, grd_b, t, eta=0.001, betta1=0.9, betta2=0.999, eps=1e-8):
        self.m_w = betta1*self.m_w + (1-betta1)*grd_w
        self.m_b = betta1*self.m_b + (1-betta1)*grd_b

        self.v_w = betta2*self.v_w + (1-betta2)*(grd_w**2)
        self.v_b = betta2*self.v_b + (1-betta2)*(grd_b**2)

        new_m_w = self.m_w / (1-betta1**t)
        new_m_b = self.m_b / (1-betta1**t)
        new_v_w = self.v_w / (1-betta2**t)
        new_v_b = self.v_b / (1-betta2**t)

        self.weights -= eta * new_m_w / (np.sqrt(new_v_w)+eps)
        self.bias -= eta * new_m_b/(np.sqrt(new_v_b)+eps)

    def forward_propagation(self, input_data):
        self.inputs = input_data
        print(self.inputs.shape)
        print(self.weights.shape)
        self.output = np.dot(self.inputs, self.weights) + self.bias
        if self.activation == 'sigmoid':
            self.output = self.sigmoid(self.output)
        if self.activation == 'relu':
            self.output = self.ReLU(self.output)
        return self.output

    def back_propagation(self, error, learning_rate, i):
        if self.activation == 'sigmoid':
            error = error * self.sigmoid_der(self.output)
        elif self.activation == 'relu':
            error = error * self.der_relu(self.output)
        grad_weights = np.dot(self.inputs.T, error)
        grad_input = np.dot(error, self.weights.T)
        self.adam(grad_weights, error.sum(axis=0), i)
        # Gradien Descent
        # self.weights -= learning_rate * grad_weights
        # self.bias -= learning_rate * error.sum(axis=0)
        return grad_input

class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def loss(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def predict(self, X_train):
        results = []
        for x in X_train:
            for layer in self.layers:
                x = layer.forward_propagation(x)
            results.append(x)

        return np.array(results)

    def call(self, X_train, y_train, lr, epoch):
        n_sample = X_train.shape[0]
        loss_history = []
        for i in range(1, epoch+1):
            input = X_train
            err = 0
            for layer in self.layers:
                input = layer.forward_propagation(input)

            err += self.loss(y_train, input)
            error = 2*(input - y_train) / len(y_train)
            for layer in reversed(self.layers):
                error = layer.back_propagation(error, lr, i)

            err /= n_sample
            print(f"Epoch {i}, Loss:", err)
            loss_history.append(err)
        return np.array(loss_history)

X, y = make_regression(n_samples=5000, n_features=5,n_informative=10 ,random_state=32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)
input_size = X_train.shape[1]
fc = DenseNetwork()
fc.add(DenseLayer(input_size, 128))
fc.add(DenseLayer(128, 256, activation='sigmoid'))
fc.add(DenseLayer(256, 70))
fc.add(DenseLayer(70, 10, activation='relu'))
fc.add(DenseLayer(10, 1))

h = fc.call(X_train, y_train, 0.01, 60)
# pred = fc.predict(X_test)
# print("My predictions:", pred)
# print("test predictions:", y_test)
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# print(lr.predict(X_test))

plt.plot(h)
plt.xlim(0, 120)
plt.ylim(0, 3)
plt.savefig("loss.png")
