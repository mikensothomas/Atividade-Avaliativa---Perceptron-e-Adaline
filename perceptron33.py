import numpy as np

data = np.array([
    [-0.3665, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
])

X = data[:, :3]

d = np.array([-1, -1, 1, 1, 1, -1, 1, -1, -1, -1])

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=5):
        self.W = np.random.rand(input_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, u):
        return np.where(u >= 0, 1, -1)

    def train(self, X, d):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                u = np.dot(X[i], self.W)
                y = self.activation(u)
                self.W += self.learning_rate * (d[i] - y) * X[i]
        return self.W

    def predict(self, X):
        u = np.dot(X, self.W)
        return self.activation(u)

input_size = X.shape[1]
perceptron = Perceptron(input_size)
weights_after_training = perceptron.train(X, d)

predictions = perceptron.predict(X)

print("Previsões:", predictions)
print("Pesos após o treinamento:", weights_after_training)
