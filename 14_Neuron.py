# Neuron
# 0.Environment build
import math
import numpy as np

# 1.neuron function
def neuron(input, weight, bias):
    output = 0
    for x,w in zip(input, weight):
        output += w*x
    output = output + bias
    output = 1 / (1 + math.exp(-output))
    return output

# 2.Instantiate
input = [1,2,3]
weight = [0.1,0.2,0.3]
bias = 0.01

output = neuron(input=input, weight=weight, bias=bias)
print(output)

# 3.Neural Network
class NN:
    def __init__(self, weight_01, bias_01, weight_02, bias_02):
        self.weight_01 = weight_01
        self.bias_01 = bias_01
        self.weight_02 = weight_02
        self.bias_02 = bias_02

    def calculate(self, x):
        output_01 = np.dot(x, self.weight_01) + self.bias_01
        output_01 = 1 / (1 + np.exp(-output_01))

        output_02 = np.dot(output_01, self.weight_02) + self.bias_02
        output_02 = 1 / (1 + np.exp(-output_02))
        return output_02

# 4.Instantiate
input_data = np.array([1,2,3])
weight_01 = np.array([[0.1,0.2,0.3],
                    [0.4,0.5,0.6],
                    [0.7,0.8,0.9]])
bias_01 = 0.05
weight_02 = np.array([[0.11,0.21,0.31],
                    [0.41,0.51,0.61],
                    [0.71,0.81,0.91]])
bias_02 = 0.06

nn = NN(weight_01=weight_01, bias_01=bias_01, weight_02=weight_02, bias_02=bias_02)
output_data = nn.calculate(input_data)
print(output_data)

# 5.Deep Neural Network
class DeepNN:
    def __init__(self, input_data, output_data, hidden_data):
        self.model = [input_data] + hidden_data + [output_data]
        self.weight = []
        self.layer_data = len(self.model)
        for i in range(self.layer_data - 1):
            current_weight = np.random.rand(self.model[i], self.model[i+1])
            self.weight.append(current_weight)
        self.bias = 0
    
    def calculate(self, data):
        for w in self.weight:
            x = np.dot(data, w)
            x = 1 / (1 + np.exp(-x))
            return x

# 6.Instantiate
input_data = 3
output_data = 1
hidden_data = [3,2,3]
deep_nn = DeepNN(input_data=input_data, output_data=output_data, hidden_data=hidden_data)
input = np.array([1,2,3])
output = deep_nn.calculate(input)
print(output)