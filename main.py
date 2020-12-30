from numpy import exp, array, random, dot


class NeuralNetwork:
    def __init__(self):
        #random.seed(1)  # Seed the random number generator, so it generates the same numbers every time the program runs.

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1 and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):  # We train the neural network through a process of trial and error, adjusting the synaptic weights each time.
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)  # Pass the training set through our neural network (a single neuron).

            error = training_set_outputs - output  # Calculate the error (The difference between the desired output and the predicted output).

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment  # Adjust the weights.

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))  # Pass inputs through our neural network (our single neuron).


if __name__ == "__main__":
    neural_network = NeuralNetwork()  # Initialise a single neuron neural network.

    #print("Random starting synaptic weights:")
    #print(neural_network.synaptic_weights)

    preTrain = neural_network.think(array([1, 0, 0]))
    print("Output before training:")
    print(preTrain)

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])  # The training set. We have 4 examples, each consisting of 3 input values and 1 output value.
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    #print("\nNew synaptic weights after training:")
    #print(neural_network.synaptic_weights)
    postTrain = neural_network.think(array([1, 0, 0]))
    print("\nConsidering new situation [1, 0, 0] -> ?:")  # Test the neural network with a new situation.
    print(postTrain)

    print("\nImprovement:")
    print(postTrain - preTrain)