# Import a random function to initialize our weights reandomly
from random import randint

# The Neural Network class
class BaseNeuralNetwork:
    # mathematical constant e
    E_CONST = 2.718281828459045
    # The rate at which our model learns
    '''
    On a general note
        Higher training rate = faster and more generic learning

        Lower training rate = more percise and accurate learning to fit our model
    '''
    TRAINING_RATE = -50
    # The topology of the network
    # This contains the ammount of layers and how many nodes in each layer via a tuple
    shape = None
    # This is used in the run method to store the intermediate values of each node
    hidden_layers = []
    # This stores the weights used in the forward and back propagation algorithm
    hidden_weights = []
    # a multidimensional (2) array for storing the bias weights
    bias_layers = []
    # Name for saving to files (Not yet implmented in this file)
    name = "tempNN"
    #
    # Initialize method
    #
    def __init__(self, name="tempNN", shape=(0, 0), loadFromFile=False):
        self.name = name        # Set name to passed in name
        self.shape = shape      # Set the shape to passed in shape
        if loadFromFile:        # decide wether to 
            self.loadWeights()  #   Load weights from a file
        else:                   # or
            self.newWeights()   #   generate random weights
    #
    # Loads weights from a file (Not yet implemented)
    #
    def loadWeights(self):
        pass
    #
    #   Generates a random weight between -1 and 1 to 2 decimal places
    #
    def randomWeight(self):
        return float(randint(0, 200)-100)/100
    #
    # Generates random weights to populate the neural network
    #
    def newWeights(self):
        # Resets all the hidden weights
        self.hidden_weights = []
        # Reset all the bias weights/layers
        self.bias_layers = []
        # for each layer of weights in our neural network
        for layer in range(0, len(self.shape)-1):
            # make a new layer of weights
            # also a matrix of 2 dimensions
            self.hidden_weights.append([])
            # for each node in the current layer i
            for i in range(self.shape[layer]):
                # make a new layer in the matrix
                self.hidden_weights[layer].append([])
                # for each node in the next layer
                for j in range(self.shape[layer+1]):
                    # populate the matrix with a new hidden weight
                    # where layer is the current weight layer
                    #   and n_layer is the current node/next layer
                    #   and l_n_layer and l_layer are the amount of 
                    #   nodes in the next layer and current layer respectively
                    #       the current matrix in index layer is size l_layerx l_n_layer 
                    self.hidden_weights[layer][i].append(self.randomWeight())
            # do that same stuff
            #   but for the bias layer
            self.bias_layers.append([])
            # this is only a 2 dimensional array, because each layer has exacly 1 bias NotADirectoryError
            #   the node has no imput, only an output of 1 and a weight
            #   which is often simplified to be an array of weights
            # for each node in the node/next layer
            for j in range(self.shape[layer+1]):
                # add a new weight/connection to the next layer to the array
                self.bias_layers[layer].append(self.randomWeight())
    #
    #   Activation method
    #
    def activate(self, x):
        ''' This is the activation method which each node applys to the output of itself
            the current one is a sigmoid function '''
        return 1/(1+pow(self.E_CONST, -x))
    #
    # Run method
    #
    def run(self, input_data):
        # make sure we were sent the correct number of inputs
        if not len(input_data) == self.shape[0]:
            return 'Error: Input Data is not in the correct format'
        # Reset the node values in the hidden layers
        self.hidden_layers = []
        # set the 1st layer of the network to out input data
        self.hidden_layers.append(list(input_data))
        # for each layer in the network
        for layer in range(len(self.shape)-1):
            # make sure there's room to store them in the list
            self.hidden_layers.append([])
            # for each node in the next layer
            for j in range(self.shape[layer+1]):
                # value is the current value of the node
                # we initialize this to the bias weight for this layer
                # because the bias is there to allow a wider domain in the activation function
                value = self.bias_layers[layer][j]
                # for each node in the current layer
                for i in range(self.shape[layer]):
                    # Multiply it's output by the weight connecting it to this node 
                    #   and add it to the running total
                    value += self.hidden_layers[layer][i] * self.hidden_weights[layer][i][j]
                # append the final value to the array
                self.hidden_layers[layer+1].append(self.activate(value))
        # Return the output layer, for debugging
        return self.hidden_layers[-1]
    #
    # Packpropagation method
    #
    def train(self, X, Y):
        self.run(X)
        bias_deltas = [[] for x in range(len(self.shape)-1)]
        previous_delta = []
        deltas = [[] for layer in range(len(self.shape)-1)]
        for weight_layer in reversed(range(len(self.shape)-1)):
            node_layer = weight_layer + 1
            if weight_layer == len(self.shape)-2:
                for k in range(self.shape[node_layer]):
                    error_output = self.hidden_layers[node_layer][k] - Y[k]
                    output = self.hidden_layers[node_layer][k]
                    output_error = error_output * output * (1 - output)
                    previous_delta.append(output_error)
                for j in range(self.shape[weight_layer]):
                    deltas[weight_layer].append([])
                    for k in range(self.shape[node_layer]):
                        delta_weight = self.TRAINING_RATE * previous_delta[k]
                        delta_weight *= self.hidden_layers[weight_layer][j]
                        deltas[weight_layer][j].append(delta_weight)
                for j in range(self.shape[node_layer]):
                    output = self.hidden_layers[node_layer][j]
                    bias_error = output * (1 - output)
                    bias_delta = bias_error * previous_delta[j] * self.TRAINING_RATE
                    bias_deltas[weight_layer].append(bias_delta)
            else:
                current_delta = []
                for j in range(self.shape[node_layer]):
                    delta_sum = 0
                    for k in range(self.shape[node_layer+1]):
                        delta_sum += previous_delta[k] * self.hidden_weights[node_layer][j][k]
                    output = self.hidden_layers[node_layer][j]
                    delta = delta_sum * output * (1 - output)
                    current_delta.append(delta)
                previous_delta = list(current_delta)
                for j in range(self.shape[node_layer]):
                    output = self.hidden_layers[node_layer][j]
                    bias_error = output * (1 - output)
                    bias_delta = bias_error * previous_delta[j] * self.TRAINING_RATE
                    bias_deltas[weight_layer].append(bias_delta)
                for i in range(self.shape[weight_layer]):
                    deltas[weight_layer].append([])
                    for j in range(self.shape[node_layer]):
                        delta_weight = self.TRAINING_RATE * current_delta[j]
                        delta_weight *= self.hidden_layers[weight_layer][i]
                        deltas[weight_layer][i].append(delta_weight)
        # Update Weights
        for layer in range(len(self.shape)-1):
            for j in range(self.shape[layer+1]):
                self.bias_layers[layer][j] += bias_deltas[layer][j]
            for i in range(self.shape[layer]):
                for j in range(self.shape[layer+1]):
                    self.hidden_weights[layer][i][j] += deltas[layer][i][j]

if __name__ == '__main__':
    NN = BaseNeuralNetwork("Miles", (2, 3, 1))
    X = [[0, 0], [1, 0], [0, 1], [1, 1]]

    Y = [[0], [1], [1], [0]]
    AMOUNT_OF_DATA = len(X)
    ITERATIONS = 100000
    for x in range(4):
        print(NN.run(X[x]))
        print(X[x])
    for x in range(ITERATIONS):
        index = randint(0, 3)
        inputData = X[index]
        outputData = Y[index]
        NN.train(inputData, outputData)
        if x%10000 == 0:
            error = 0
            for y in range(AMOUNT_OF_DATA):
                NN.run(X[y])
                error += abs(Y[y][0] - NN.hidden_layers[-1][0])
            print("Raw value: " + str(error)[:4])
            print("Error: " + str(float(error/0.04))[:4] + "%")
    for x in range(AMOUNT_OF_DATA):
        print(NN.run(X[x]))
        print(X[x])
