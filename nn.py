import numpy as np
from nodes import Layer

class NeuralNetwork(object):
    """A class to store and operate a neural network. Calls upon the
    nodes class which defines the node and layer level operations. For
    an example of initialising and using the classes see
    nn_wrapper.py

    Parameters for initialization
    -----------------------------
    numberParameters : int
        The number of input layer nodes, or conversely, the number of
        different parameters defined for each data point which the
        neural network will select on.
    numberHiddenNodes : int
        The number of nodes in the hidden layer. As a rough initial
        guess this should be at least as great as the number of
        parameters. This class will automatically prune nodes below a
        certain threshold, which mitigates the penalties of setting
        this number too high. Default value is here set to 11.
    numberOutputNodes : int
        The number of nodes in the output layer, or the number of
        different classifications to fit the data to. Default value is
        here set to 1, which is sufficient for classifying a source as
        signal or noise.

    Parameters for training
    -----------------------
    trainingSet : 2D-float array
        The data to train the neural network on. Each row is a data
        point with a column for each parameter
    targets : float array
        The target value for each data point in the training set that
        the output nodes should output. Each row is a target value,
        multiple columns are used for multiple output nodes.
    maximumIterations : int
        The maximum number of times the neural network should iterate
        on the training data. Default is here set to 100.
    stepSize : float
        A threshold for the weight increase. If at any iteration the
        change in all weights drops below this threshold the training
        stops. Generally, this shouldn't be used as the network
        training might well plateau for awhile, making small changes,
        before falling into a minimum and improving the performance
        again. Default is 0.000001
    reset : Boolean
        If True, the network will clean out the training data from the
        neural network after the final iteration so that the instance
        can be saved with just the weights (and hence a small file
        size). Default is True
    outputFile : string
      The name of the output file to store information on the state of
      the solution at each iteration. Each row of this ASCII file
      records the iteration number, the mean difference between the
      target and training data, the median difference, the standard
      deviation, the minimum difference, and the maximum
      difference. The default file name is "nn_output.dat"
    """

    def __init__(self, numberParameters, numberHiddenNodes=11, numberOutputNodes=1):
        # Set global parameters
        self.numberHiddenNodes = numberHiddenNodes
        self.numberOutputNodes = numberOutputNodes
        self.numberParameters = numberParameters
        # Initialize hidden and output layers
        print "Initializing weights"
        self.hiddenLayer = Layer(numberHiddenNodes, numberParameters, type="hidden")
        self.outputLayer = Layer(numberOutputNodes, numberHiddenNodes, type="output")

    def forwardPropagate(self,data):
        # Run the input data forwards through the neural network
        # First update the hidden nodes
        output = self.hiddenLayer.forwardPropagate(data)
        # and then update the output nodes
        output = self.outputLayer.forwardPropagate(output)
        # Return the final classifications of the data
        return output[:,0]

    def pruneNodes(self, threshold=0.001):
        # Look at the weights assigned to each hidden layer node by
        # each output layer node. If the maximum weight assigned to a
        # hidden layer node drops below the value of threshold, then
        # remove it. Returns a boolean indicating if any node was
        # removed.
        nodeWeights = np.max([np.abs(node.weights[:-1]) for node in self.outputLayer.nodes],axis=0)
        sel = (nodeWeights <  threshold)
        if (sel.any() == True):
            for i, flag in enumerate(sel):
                if (flag):
                    print "Node {} has dropped below significance and is being pruned".format(i)
                    self.hiddenLayer.prune(i, self.outputLayer)
            return True
        else:
            return False

    def updateWeights(self, newWeights):
        self.hiddenLayer.updateWeights(newWeights[:self.numberParameters])
        self.outputLayer.updateWeights(newWeights[self.numberParameters:])

    def emptyData(self):
        # Call layer functions to clear out all the stored data.
        self.hiddenLayer.emptyData()
        self.outputLayer.emptyData()

    def trainNeuralNetwork(self, trainingSet, targets, maximumIterations=100, stepSize=0.000001, clean=True, outputFile="nn_output.dat"):
        # Train the neural network on these datasets. 
        # Initialize counters
        iterations = 0
        step = 1000
        newOutputWeights,newHiddenWeights = [],[]

        print "Training Neural Network"

        while (step > stepSize) & (iterations < maximumIterations):
            # Calculate and average updated weights for the training sets
            print "iteration",iterations
            output = self.forwardPropagate(trainingSet)
            # Calculate metrics for solution
            diff=output-targets
            absdiff=np.abs(diff)
            mean=np.mean(absdiff)
            median=np.median(absdiff)
            std=np.std(absdiff)
            mindiff=np.min(diff)
            maxdiff=np.max(diff)
            print "Deviations from target classifications: mean: {}, median: {}, standard deviation {}, minimum difference: {}, maximum difference {}".format(mean,median,std,mindiff,maxdiff)
            with open(outputFile,'a') as outfile:
                outfile.write('{:02d}, {}, {}, {}, {}, {}\n'.format(iterations, mean, median, std, mindiff, maxdiff))
            # Calculate weight updates for each layer
            newOutputWeights = self.outputLayer.calculateWeightUpdates(self.hiddenLayer, targets=targets)
            newHiddenWeights = self.hiddenLayer.calculateWeightUpdates(self.outputLayer, data=trainingSet)
            stepOutput = np.max(np.abs(newOutputWeights[:-1]))
            stepHidden = np.max(np.abs(newHiddenWeights[:-1]))
            step = np.max(stepOutput, stepHidden)
            # Apply weight updates
            self.outputLayer.updateWeights(newOutputWeights)
            self.hiddenLayer.updateWeights(newHiddenWeights)
            # Check if any nodes need to be pruned
            self.pruneNodes()
            iterations+=1

        if (reset):
            # Clean out data from neural network just leaving weights
            self.emptyData()
