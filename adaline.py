import numpy as np

data = np.array([
    [-0.6508, 0.1097, 4.0009, -1],
    [-1.4492, 0.8896, 4.4005, -1],
    [2.0850, 0.6876, 12.0710, -1],
    [0.2626, 1.1476, 7.7985, -1],
    [0.6418, 1.0234, 7.0427, 1],
    [0.2569, 0.6730, 8.3265, -1],
    [1.1155, 0.6043, 7.4446, 1],
    [0.0914, 0.3399, 7.0677, -1],
    [0.0121, 0.5256, 4.6316, 1],
    [-0.0429, 0.4660, 5.4323, -1],
    [0.4340, 0.6870, 8.2287, -1],
    [0.2735, 1.0287, 7.1934, 1],
    [0.4839, 0.4851, 7.4850, -1],
    [0.4089, -0.1267, 5.5019, -1],
    [1.4391, 0.1614, 8.5843, 1],
    [-0.9115, -0.1973, 2.1962, -1],
    [0.3654, 1.0475, 7.4858, 1],
    [0.2144, 0.7515, 7.1699, 1],
    [0.2013, 1.0014, 6.5489, 1],
    [0.6483, 0.2183, 5.8991, 1],
    [-0.1147, 0.2242, 7.2435, -1],
    [-0.7970, 0.8795, 3.8762, 1],
    [-0.1617, 0.2550, 9.0275, -1],
    [0.5369, 0.5157, 5.3665, 1],
    [0.1962, 0.0000, 6.4422, -1],
    [1.2013, 0.4939, 8.2117, 1],
    [1.3957, 0.4720, 9.6072, 1],
    [0.9929, 0.2246, 6.7101, 1],
    [0.2019, 0.6192, 7.9293, -1],
    [0.2012, 0.2611, 5.4631, 1]

    # [-0.3665,	0.0620,	5.9891],
    # [-0.7842,	1.1267,	5.5912],
    # [0.3012,	0.5611,	5.8234],
    # [0.7757,	1.0648,	8.0677],
    # [0.1570,	0.8028,	6.3040],
    # [-0.7014,	1.0316,	3.6005],
    # [0.3748,	0.1536,	6.1537],
    # [-0.6920,	0.9404,	4.4058],
    # [-1.3970,	0.7141,	4.9263],
    # [-1.8842,	-0.2805, 1.2548]
])

X = data[:, :3]
d = data[:, 3]

class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=5):
        self.W = np.random.rand(input_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, d):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                u = np.dot(X[i], self.W)
                error = d[i] - u
                self.W += self.learning_rate * error * X[i]
        return self.W

    def predict(self, X):
        u = np.dot(X, self.W)
        return np.where(u >= 0, 1, -1)

input_size = X.shape[1]

adaline = Adaline(input_size)
weights_after_training_adaline = adaline.train(X, d)

predictions_adaline = adaline.predict(X)

print("Previsões do Adaline:", predictions_adaline)
print("Pesos após o treinamento do Adaline:", weights_after_training_adaline)
