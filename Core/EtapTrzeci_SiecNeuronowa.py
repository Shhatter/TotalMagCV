import json
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

'''
postTestLight_Feature
postTestLight_Labels

postTestNeg_Feature
postTestNeg_Labels

postTestHard_Feature
postTestHard_Labels

pos_Feature
pos_labels
neg_Feature
neg_Labels
'''


def fetchData(filepath):
    with open(filepath, "r")as filehandler:
        rawData = np.array(json.load(filehandler))
        print("Data from file: %s total dataset size = %d" % (filepath, len(rawData)))
        length = len(rawData[0]) - 1
        features = rawData[:, range(0, length)]
        labels = rawData[:, length]
        labels = labels.reshape((len(rawData), 1))

    return features, labels


pos_Feature, pos_labels = fetchData("ExposedData_Sick_TOTAL.txt")

neg_Feature, neg_Labels = fetchData("ExposedData_Healthy_TOTAL.txt")
# print(neg_Labels)
x_data = np.concatenate((pos_Feature, neg_Feature), axis=0)
y_data = np.concatenate((pos_labels, neg_Labels), axis=0)
# print(y_data)


postTestHard_Feature, postTestHard_Labels = fetchData("ExposedData_Sick_TOTAL_test.txt")

postTestNeg_Feature, postTestNeg_Labels = fetchData("ExposedData_Healthy_TOTAL_test.txt")

x_data_test = np.concatenate((postTestHard_Feature, postTestNeg_Feature), axis=0)
y_data_test = np.concatenate((postTestHard_Labels, postTestNeg_Labels), axis=0)

'''





'''

n_nodes_hl1 = 70
n_nodes_hl2 = 50
n_nodes_hl3 = 30
n_classes = 1

batch_size = 92
hm_epochs = 10

x = tf.placeholder('float', [None, 92])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([92, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights'] + output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_data) / batch_size)
            x_batches = np.array_split(x_data, total_batch)
            y_batches = np.array_split(y_data, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y
                                })
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: x_data_test, y: y_data_test}))


train_neural_network(x)

'''
training_epochs = 500
learning_rate = 0.0001
batch_size = 100
display_step = 1

n_input = 92
n_hidden_1 = 70
n_hidden_2 = 30
n_classes = 1

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}









def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    
'''
