import numpy as np
from photometry import Photometry

class Node(object):
    """A class to store the variables and methods for a given node in
    the neural network.

    Parameters
    ----------
    numberWeights : int
        The number of weights assigned to this node.
    data : float array
        The data to pass through the node. Each row is a distinct data
        point with the columns recording the parameters for that
        datum.
    weightUpdates : float array
        The updates to the nodes weights
    """

    def __init__(self, numberWeights):
        # Randomize the weights
        self.weights = np.random.random(numberWeights+1)
        # Multiply the weight of the bias node by the number of other
        # nodes such that the combined weights will be close to zero.
        self.weights[-1]*=numberWeights
        self.output = 0.0
        self.errors = 0.0

    def calculateOutput(self, data):
        # Calculate the output from each node by: summing the product
        # of each weight and input, subtracting the bias weight, and
        # then passing the value through a sigmoid function.
        numberDataPoints = np.shape(data)[0]
        output = data * self.weights[:-1]
        output = np.sum(output, axis=1) - self.weights[-1] 
        # Return an array of activations for each data point
        self.output = 1.0 / (1.0 + np.exp(-output)) 

    def updateWeights(self, weightUpdates):
        self.weights += weightUpdates
        

class HiddenNode(Node):
    """An instance of a node. Takes array of variables output by
    InputNodes for each datum. Combines and outputs them as the value
    of a threshold function.

    Parameters
    ----------
    numberWeights : int
        As defined in the Node Class
    outputLayer : Layer
        The layer these hidden nodes output to
    nodeNumber : int
        The number assigned to this node in the output layer
    data : float array
        As defined in the Node class
    learningRate : float
        As defined in the Layer class
    """
    def __init__(self, numberWeights): 
        Node.__init__(self, numberWeights)

    def calculateErrorFunctions(self, outputLayer, nodeNumber):
        # Calculate the error functions, defined as the
        # differentiation of the sigmoid function multiplied by the
        # error functions of the output layers which are passed backwards
        errorFunctions = [node.errors for node in outputLayer.nodes]
        weights = [node.weights[nodeNumber] for node in outputLayer.nodes]
        if (np.shape(errorFunctions)[0]) == 1:
            self.errors = self.output * (1 - self.output) * (errorFunctions[0] * weights[0])
        else:
            errorWeights = [errorFunctions[i]*weights[i] for i in range(np.shape(errorFunctions)[0])]
            self.errors = self.output * (1 - self.output) * np.sum(errorWeights, axis=0)

    def calculateWeightUpdates(self, data, learningRate):
        # Returns weight updates of each hidden layer node averaged
        # across all data values
        weightAdditions = learningRate * self.errors * np.asfarray(data).transpose()
        biasAdditions = np.mean(learningRate * self.errors * -1)
        average = np.mean(weightAdditions, axis=1)
        return np.append(average, biasAdditions)
    

class OutputNode(Node):
    """An instance of a node. Takes array of variables output by
    HiddenNodes for each datum. Combines and outputs them as a final
    value indicating the likelihood of the classification.

    Parameters
    ----------
    numberWeights : int
        As defined in the Node Class
    targets : float
        The target values for the output nodes
    hiddenLayer : Layer
        The layer that inputs into these nodes 
    learningRate : float
        As defined in the Layer class
    """
    def __init__(self, numberWeights): 
        Node.__init__(self, numberWeights)

    def calculateErrorFunctions(self, targets):
        # Calculate the error functions, defined as the
        # differentiation of the sigmoid function multiplied by the
        # difference between the output and the target
        self.errors = self.output * (1 - self.output) * (targets - self.output)

    def calculateWeightUpdates(self, hiddenLayer, learningRate):
        # Returns weight updates for each output layer node averaged
        # across all data values
        hiddenLayerResponse=[node.output for node in hiddenLayer.nodes]
        weightAdditions = learningRate * self.errors * hiddenLayerResponse
        biasAdditions = np.mean(learningRate * self.errors * -1) 
        average = np.mean(weightAdditions, axis=1)
        return np.append(average, biasAdditions)


class Layer(object):
    """A class for storing a layer, which comprises a collection of
    nodes. Has to explicity check if it is a hidden layer or an output
    layer as python doesn't support method overloading :(

n instance of a node. Takes array of variables output by
    HiddenNodes for each datum. Combines and outputs them as a final
    value indicating the likelihood of the classification.

    Parameters
    ----------
    numberNodes : int
        The number of nodes in this layer
    numberWeights : int
        The number of weights for each node in this layer
    type : string
        The type of layer. Can either be "hidden" or "output"
    data : float array
        The data to pass through the node. Each row is a distinct data
        point with the columns recording the parameters for that
        datum.
    layer : Layer
        Pointer to the other layer in the neural network
    targets : float
        The target values for the output nodes
    hiddenLayer : Layer
        The layer that inputs into these nodes 
    learningRate : float
        The efficiency of the weight updates. Each calculated update
        is multiplied by this value. It can be set small to avoid
        overshooting a solution, or large to speed up optimization. A
        future update might automatically decrease the size of this
        number for successive iterations. Default is 0.3.
    nodeNumber : int
        The number of the node to prune from the hidden layer
    outputLayer : Layer
        The output layer
    weightUpdates : float array
        The updates to the weights for each node in the layer
    """
    def __init__(self, numberNodes, numberWeights, type="hidden"):
        # Initialize layers
        if (type=="hidden"):
            self.type = type
            self.nodes = [HiddenNode(numberWeights) for i in range(numberNodes)]
        elif (type=="output"):
            self.type = type
            self.nodes = [OutputNode(numberWeights) for i in range(numberNodes)]
        else:
            print "Layer type not recognized. Valid options are 'hidden' and 'output'"
        
    def forwardPropagate(self,data):
        # Calculate and return output for each node in layer
        for node in self.nodes:
            node.calculateOutput(data)
        return np.asfarray([node.output for node in self.nodes]).transpose()

    def calculateWeightUpdates(self, layer, data=0, targets=0, learningRate=0.3):
        # Calculate and return the weight updates for each node in layer
        newWeights=[]
        #for name,value in kwargs.items():
        #    if name=='learningRate': learningRate=value
        for i,node in enumerate(self.nodes):
            if (self.type == "hidden"):
                node.calculateErrorFunctions(layer, i)
                weightUpdates = node.calculateWeightUpdates(data,learningRate)
            else:
                node.calculateErrorFunctions(targets)
                weightUpdates = node.calculateWeightUpdates(layer,learningRate)
            if (len(newWeights)==0):
                newWeights=weightUpdates
            else:
                newWeights=np.column_stack([newWeights,weightUpdates])
        return newWeights.transpose()

    def prune(self,nodeNumber,outputLayer):
        # Remove a node from the hidden layer and references to it in
        # the output layer. Note, python takes care of array bounds
        nodeArray = self.nodes[:nodeNumber] + self.nodes[nodeNumber+1:]
        self.nodes = nodeArray

        for node in outputLayer.nodes:
            weightsArray = np.append(node.weights[:nodeNumber], node.weights[nodeNumber+1:])
            errorsArray  = np.append(node.errors[:nodeNumber], node.errors[nodeNumber+1:])
            node.weights = weightsArray
            node.errors = errorsArray

    def emptyData(self):
        # Set the data and error functions to empty lists for each node
        for node in self.nodes:
            node.output = []
            node.errors = []

    def updateWeights(self, weightUpdates):
        # Apply weight updates for each node
        for i,node in enumerate(self.nodes):
            node.updateWeights(weightUpdates[i])

    
