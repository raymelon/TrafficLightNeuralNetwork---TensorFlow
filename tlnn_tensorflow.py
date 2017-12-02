import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()


## 1. setup neural network component sizes
# 2 input neurons
inputs = tf.placeholder(tf.float32, shape=[None, 2])

# 1 output neurons
outputs = tf.placeholder(tf.float32, shape=[None, 1])

# 2 hidden neurons
hidden_neurons = 2


## 2. initialize neural network component values
# input layer
weights_input = tf.Variable(tf.truncated_normal([2, hidden_neurons]))
biases_input = tf.Variable(tf.zeros([hidden_neurons]))

# hidden layer
weights_hidden = tf.Variable(tf.truncated_normal([hidden_neurons, 2]))
biases_hidden = tf.Variable(tf.zeros([2]))

# output layer
weights_output = tf.Variable(tf.truncated_normal([2, 1]))
biases_output = tf.Variable(tf.zeros([1]))


## 3. setup neural network functions
# activation function
input_activation = tf.nn.sigmoid(tf.matmul(inputs, weights_input) + biases_input)
hidden_activation = tf.nn.sigmoid(tf.matmul(input_activation, weights_hidden) + biases_hidden)
output_activation = tf.nn.sigmoid(tf.matmul(hidden_activation, weights_output) + biases_output)

# optimizer
error = 0.5 * tf.reduce_sum(tf.subtract(output_activation, outputs) * tf.subtract(output_activation, outputs))
learning_rate = 0.005
step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)


## 4. build the network
session.run(tf.initialize_all_variables())


## 5. train
training_size = 300
training_inputs = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.1]] * training_size
training_outputs = [[0.3], [0.4], [0.5], [0.6], [0.1], [0.2]] * training_size

epochs = 20000

for epoch in range(epochs):
	_, error_rate = session.run(	[step, error],
									feed_dict = 
									{ 	inputs: np.array(training_inputs),
										outputs: np.array(training_outputs)
									}
								)

	print('epoch: ' + str(epoch) + ' | error: ' + str((error_rate * 100)) + '%')


## 6. test

print(session.run(output_activation, feed_dict = {inputs: np.array([[0.1, 0.2]])}))
print(session.run(output_activation, feed_dict = {inputs: np.array([[0.2, 0.3]])}))
print(session.run(output_activation, feed_dict = {inputs: np.array([[0.3, 0.4]])}))
print(session.run(output_activation, feed_dict = {inputs: np.array([[0.4, 0.5]])}))
print(session.run(output_activation, feed_dict = {inputs: np.array([[0.5, 0.6]])}))
print(session.run(output_activation, feed_dict = {inputs: np.array([[0.6, 0.1]])}))
