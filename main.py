import numpy as np
from numpy import array, random, dot, tanh

monday = input("Who came Monday? \n Do Ludovic, Jonathan, Étienne \nEnter three space-separated values: ")
monday_Matrix = np.array(list(map(int, monday.split())))

Tuesday = input("Who came Tuesday? \n Do Ludovic, Jonathan, Étienne \nEnter three space-separated values: ")
Tuesday_Matrix = np.array(list(map(int, Tuesday.split())))

Wednesday = input("Who came Wednesday? \n Do Ludovic, Jonathan, Étienne \nEnter three space-separated values: ")
Wednesday_Matrix = np.array(list(map(int, Wednesday.split())))

Thursday = input("Who came Thursday \n Do Ludovic, Jonathan, Étienne,  \nEnter three space-separated values: ")
Thursday_Matrix = np.array(list(map(int, Thursday.split())))

Vincent_action = input("When did Vincent come to class? \nEnter four space-separated values: ")
Vincent_Matrix = np.array(list(map(int, Vincent_action.split())))

print('We need to determine if Vincent will go to its class')
user_input = input("Enter three space-separated values: ")
w = np.array(list(map(int, user_input.split())))


class NeuralNetwork:

    def __init__(self):
        random.seed(1)
        self.weight_matrix = 2 * random.random((3, 1)) - 1

    def tanh(self, x):
        return tanh(x)

    # derivative of tanh function.
    # Needed to calculate the gradients.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2

    # forward propagation
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

    # training the neural network.
    def train(self, train_inputs, train_outputs,
              num_train_iterations):
        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)

            # Calculate the error in the output.
            error = train_outputs - output

            # multiply the error by input and then
            # by gradient of tanh function to calculate
            # the adjustment needs to be made in weights
            adjustment = dot(train_inputs.T, error *
                             self.tanh_derivative(output))

            # Adjust the weight matrix
            self.weight_matrix += adjustment


# Driver Code
if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print('Random weights at the start of training')
    print(neural_network.weight_matrix)

    train_inputs = array([monday_Matrix, Tuesday_Matrix, Wednesday_Matrix, Thursday_Matrix])
    train_outputs = array([Vincent_Matrix]).T

    neural_network.train(train_inputs, train_outputs, 10000)

    print('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    print(neural_network.forward_propagation(array(w)))

import turtle


def draw_circle(x, y, radius, text):
    turtle.speed(1)
    turtle.penup()
    turtle.goto(x, y - radius)
    turtle.pendown()
    turtle.circle(radius)
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.write(text, align="center", font=("Arial", 8, "normal"))


def draw_line(start, end):
    turtle.speed(1)
    turtle.penup()
    turtle.goto(start)
    turtle.pendown()
    turtle.goto(end)


def draw_perceptron():
    turtle.speed(1)

    # Draw input circles
    draw_circle(-150, 150, 20, 'Ludovic')
    draw_circle(-150, 50, 20, 'Jonathan')
    draw_circle(-150, -50, 20, 'Étienne')

    row_index = 0
    col_index = 0
    w1 = neural_network.weight_matrix[row_index, col_index]

    row_index = 1
    col_index = 0
    w2 = neural_network.weight_matrix[row_index, col_index]

    row_index = 2
    col_index = 0
    w3 = neural_network.weight_matrix[row_index, col_index]

    draw_circle(-30, 100, 20, w1)
    draw_circle(-30, 50, 20, w2)
    draw_circle(-30, 0, 20, w3)

    # Draw sum circle
    draw_circle(80, 50, 20, 'Sum')

    # Draw sigma circle (activation)
    draw_circle(160, 50, 20, 'Σ')

    # Draw output circle
    draw_circle(260, 50, 20, neural_network.forward_propagation(array(w)))

    # Connect circles with lines
    draw_line((-130, 150), (-50, 100))
    draw_line((-130, 50), (-50, 50))
    draw_line((-130, -50), (-50, 0))

    draw_line((-10, 100), (60, 50))
    draw_line((-10, 50), (60, 50))
    draw_line((-10, 0), (60, 50))

    draw_line((100, 50), (140, 50))
    draw_line((180, 50), (240, 50))
    turtle.penup()

    turtle.goto(0, -100)  # Move the turtle to (-100, -100)
    turtle.pendown()

    if (w1 > 1 and w2 > 1) or (w1 > 1 and w3 > 1) or (w2 > 1 and w3 > 1):
        turtle.write('Not enough data', align="center", font=("Arial", 15, "bold"))
    else:
        if neural_network.forward_propagation(array(w)) > 0.9:
            turtle.write('Vincent will go to the class', align="center", font=("Arial", 15, "bold"))
        else:
            turtle.write('Vincent will not go to the class', align="center", font=("Arial", 15, "bold"))

    turtle.done()


if __name__ == "__main__":
    draw_perceptron()
