import numpy as np
from numpy import array, random, dot, tanh


print('This is a questionnaire to determine if you should invest money based on your situation.\nI will ask you three questions. You only have to answer using 1 for yes and 0 for no.')

w = []

# Ask the first question
question1 = "You should only invest money that you are ready to lose.\nDo you have money you can afford to lose?\nPress [1] + [enter] if yes or press [0] + [enter] if no. "
answer1 = input(question1)
w.append(int(answer1))

# Ask the second question
question2 = "Do you Have an Emergency Fund\nPress [1] + [enter] if yes or press [0] + [enter] if no. "
answer2 = input(question2)
w.append(int(answer2))

# Ask the third question
question3 = "Do you Have Long-Term Financial Goals\nPress [1] + [enter] if yes or press [0] + [enter] if no. "
answer3 = input(question3)
w.append(int(answer3))

# Convert the list 'w' to a NumPy array
w = np.array(w)

'''user_input = input("You should only invest money that you are ready to lose \nDo you have money you can afford to lose?\n\nPress [1] + [enter] if yes or press [0] + [enter] if no. ")
w = np.array(list(map(int, user_input.split())))'''

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

    training_set_inputs = array([[0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
    training_set_outputs = array([[0, 1, 0, 0, 0, 1, 1]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    print(neural_network.forward_propagation(array(w)))

import turtle


def draw_circle(x, y, radius, text):
    turtle.speed(3)
    turtle.penup()
    turtle.goto(x, y - radius)
    turtle.pendown()
    turtle.circle(radius)
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.write(text, align="center", font=("Arial", 8, "normal"))


def draw_line(start, end):
    turtle.speed(3)
    turtle.penup()
    turtle.goto(start)
    turtle.pendown()
    turtle.goto(end)


def draw_perceptron():
    turtle.speed(3)

    # Draw input circles
    draw_circle(-150, 150, 20, 'Readiness to lose money')
    draw_circle(-150, 50, 20, 'Time horizon')
    draw_circle(-150, -50, 20, 'y')

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
        turtle.write('It is advisable to further assess your financial\nsituation and investment knowledge before making a decision.', align="center", font=("Arial", 15, "bold"))
    else:
        if neural_network.forward_propagation(array(w)) > 0.9:
            turtle.write('The person who answered the questions is ready to invest', align="center", font=("Arial", 15, "bold"))
        else:
            turtle.write('The person who answered the questions is not ready to invest', align="center", font=("Arial", 15, "bold"))

    turtle.done()


if __name__ == "__main__":
    draw_perceptron()
