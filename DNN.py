import math
import random
import numpy as np
from copy import deepcopy

class Activation(object):
    def fn(x):
        pass

    def fn_(x):
        pass

class ReLU(Activation):
    def fn(x):
        return 0 if x < 0 else x
    def fn_(x):
        return 0 if x < 0 else 1 if x > 0 else np.nan

class Sigmoid(Activation):
    def fn(x):
        return 1.0 / (1 + np.exp(-x))
    def fn_(x):
        return x * (1 - x)

class NoActivation(Activation):
    def fn(x):
        return x
    def fn_(x):
        return 1

class GeneticOptimizer(object):
        def __init__(self, X, y, dnn, maxError = 10e-3, initRange = (-1, 1), \
                     maxGeneration = 100, populationSize = 1000, initPopulationSize = 1000, selectTopPopulation = 100):
            self.MAE = 1
            self.MSE = 2

            self.X = X
            self.y = y
            self.dnn = dnn
            self.maxError = maxError
            self.initRange = initRange
            self.maxGeneration = maxGeneration
            self.populationSize = populationSize
            self.initPopulationSize = initPopulationSize
            self.selectTopPopulation = selectTopPopulation

            self.varRange = initRange
            self.bestMSE = +np.inf
            self.bestMSEGen = -1
            self.varChangeRate = 10
            self.waitForBetterResult = 0
            self.lyrGeneration = 0
            self.lyrPopulation = []

            self.initialization()
            while True:
                self.evaluation()
                if self.termination() == True: 
                    break
                self.selection()
                self.variation()

        def initialization(self):
            for i in range(self.initPopulationSize):
                newLayers = deepcopy(self.dnn.layers)
                for layer in newLayers:
                    layer.weights = np.random.uniform(self.initRange[0], self.initRange[1], size = layer.weights.shape)
                    layer.biases = np.random.uniform(self.initRange[0], self.initRange[1], size = layer.biases.shape)
                self.lyrPopulation.append((newLayers, -1))

        def evaluation(self):
            for i, item in enumerate(self.lyrPopulation):
                if item[1] == -1:
                    self.dnn.layers = item[0]
                    y_ = self.dnn.predict(self.X)
                    mae, mse = self.dnn.loss(y_, self.y)
                    self.lyrPopulation[i] = (item[0], mae, mse)
            self.lyrPopulation.sort(key=lambda x: x[self.MSE])

        def termination(self):
            print(f"[INFO] Generation={self.lyrGeneration}, Search Range={self.varRange}, MAE={self.lyrPopulation[0][self.MAE]}, MSE={self.lyrPopulation[0][self.MSE]}")
            if self.lyrPopulation[0][self.MSE] < self.maxError or self.lyrGeneration == self.maxGeneration:
                return True

        def selection(self):
            self.lyrPopulation = self.lyrPopulation[:self.selectTopPopulation]

        def variation(self):
            if self.lyrPopulation[0][self.MSE] < self.bestMSE:
                self.bestMSE = self.lyrPopulation[0][self.MSE]
                self.bestMSEGen = self.lyrGeneration
            elif self.waitForBetterResult == 10: # Wait 10 times for better result
                self.waitForBetterResult = 0
                self.varRange = (self.varRange[0] / self.varChangeRate, self.varRange[1] / self.varChangeRate)
            else:
                self.waitForBetterResult += 1
            
            newLyrPopulation = []
            for i in range(math.floor(self.populationSize / self.selectTopPopulation)):
                for layers in self.lyrPopulation:
                    if i == 0: # Keep a clean copy of top populations of prev gen just in case...
                        newLyrPopulation.append(layers)
                    else:
                        newLayers = deepcopy(layers[0])
                        for layer in newLayers:
                            weightsRandomChange = np.random.uniform(self.varRange[0], self.varRange[1], size = layer.weights.shape)
                            biasesRandomChange = np.random.uniform(self.varRange[0], self.varRange[1], size = layer.biases.shape)
                            layer.weights = layer.weights + weightsRandomChange
                            layer.biases = layer.biases + biasesRandomChange
                        newLyrPopulation.append((newLayers, -1))
            self.lyrGeneration += 1
            self.lyrPopulation = newLyrPopulation

class DNN(object):
    class Layer(object):
        def __init__(self, units, activation, inputShape):
            self.units = units
            self.activation = activation
            self.inputShape = inputShape
            self.weights = np.zeros((self.units, self.inputShape))
            self.biases = np.zeros((self.units, 1))
            self.netOuts = np.zeros((self.units, 1))
            self.finalOuts = np.zeros((self.units, 1))

    def __init__(self):
        self.layers = []
    
    def addDenseLayer(self, units, activation, inputShape = None):
        inputShape = self.layers[-1].units if inputShape == None else inputShape
        layer = self.Layer(units, activation, inputShape)
        self.layers.append(layer)


    def feedforward(self, X):
        X = np.reshape(X, (len(X), 1))
        for layer in self.layers:
            layer.netOuts = layer.weights @ X + layer.biases
            X = layer.finalOuts = np.reshape(np.fromiter((layer.activation.fn(x) for x in layer.netOuts), float), layer.netOuts.shape)
        return X

    def predict(self, X):
        y_ = np.zeros((len(X), 1))
        for i, x in enumerate(X):
            y_[i] = self.feedforward(x)
        return y_

    def loss(self, y_, y):
        res = y_ - y
        mae = np.sum(abs(res)) / len(y)
        mse = 0.5 * np.sum((res) ** 2) / len(y)
        return mae, mse

    def fit(self, X, y):
        populationSize = 10
        initPopulationSize = 100
        selectTopPopulation = math.floor(populationSize / 10)
        optimizar = GeneticOptimizer(X, y, self, maxError = 0.9, initRange = (-1, 1), \
                                    maxGeneration = -1, populationSize = populationSize, initPopulationSize = initPopulationSize, selectTopPopulation = selectTopPopulation)
        self.layers = optimizar.lyrPopulation[0][0]

FEATURE_SIZE = 2
TRAINING_SIZE = 1000
EVALUATION_SIZE = 100
DATASET_SIZE = TRAINING_SIZE + EVALUATION_SIZE
ds = np.zeros( (DATASET_SIZE, FEATURE_SIZE + 1) )
for i in range(DATASET_SIZE):
    x1 = random.uniform(-10, 10)
    x2 = 3 #random.uniform(1, 10)
    ds[i][0] = x1
    ds[i][1] = x2
    ds[i][2] = x1 ** x2

X_train = ds[0:TRAINING_SIZE, 0:FEATURE_SIZE]
y_train = ds[0:TRAINING_SIZE, FEATURE_SIZE:FEATURE_SIZE + 1]
X_eval = ds[TRAINING_SIZE:DATASET_SIZE, 0:FEATURE_SIZE]
y_eval = ds[TRAINING_SIZE:DATASET_SIZE, FEATURE_SIZE:FEATURE_SIZE + 1]

dnn = DNN()
dnn.addDenseLayer(10, ReLU, FEATURE_SIZE)
dnn.addDenseLayer(5, ReLU)
dnn.addDenseLayer(5, ReLU)
dnn.addDenseLayer(1, NoActivation)
dnn.fit(X_train, y_train)

y_train_predict = dnn.predict(X_train)
y_eval_predict = dnn.predict(X_eval)
for i in range(len(X_train)):
    print(f"{i}. Training: X = ", X_train[i], "y = ", y_train[i], "y_ = ", y_train_predict[i])
print(f"Training MAE and MSE = {dnn.loss(y_train_predict, y_train)}")
for i in range(len(X_eval)):
    print(f"{i}. Evalting: X = ", X_eval[i], "y = ", y_eval[i], "y_ = ", y_eval_predict[i])
print(f"Evalting MAE and MSE = {dnn.loss(y_eval_predict, y_eval)}")

# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn import datasets

# print("[INFO] loading MNIST (sample) dataset...")
# digits = datasets.load_digits()
# data = digits.data.astype("float")
# data = (data - data.min()) / (data.max() - data.min())
# print("[INFO] samples: {}, dim: {}".format(data.shape[0],
# 	data.shape[1]))
# # construct the training and testing splits
# (trainX, testX, trainY, testY) = train_test_split(data,
# 	digits.target, test_size=0.25)
# print(trainX.shape)
# # convert the labels from integers to vectors
# trainY = LabelBinarizer().fit_transform(trainY)
# testY = LabelBinarizer().fit_transform(testY)
# # train the network
# print("[INFO] training network...")
# dnn = DNN()
# dnn.addDenseLayer(trainX.shape[1], Sigmoid, 28*28)
# dnn.addDenseLayer(32, Sigmoid)
# dnn.addDenseLayer(16, Sigmoid)
# dnn.addDenseLayer(10, NoActivation)
# dnn.fit(trainX, trainY)

