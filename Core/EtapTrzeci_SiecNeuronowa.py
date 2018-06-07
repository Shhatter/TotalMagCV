import json
import tensorflow as tf
import numpy as np

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
        print("Data fom file: %s total dataset size = %d" % (filepath, len(rawData)))
        length = len(rawData[0]) - 1
        features = rawData[:, range(0, length)]
        labels = rawData[:, length]
        labels = labels.reshape((len(rawData), 1))

    return features, labels


def extractData(typeOfData):
    global positiveData, negativeData, positiveDataTest, negativeDataTest
    if typeOfData == "total":
        with open("ExposedData_Sick_TOTAL.txt", "r")as filehandle:
            positiveData = json.load(filehandle)

        print(len(positiveData))
        print(positiveData)

        with open('ExposedData_Healthy_TOTAL.txt', 'r') as filehandle:
            negativeData = json.load(filehandle)
        print(len(positiveData))
        print(negativeData)

        with open("ExposedData_Healthy_TOTAL_test.txt", "r")as filehandle:
            positiveData = json.load(filehandle)

        print(len(positiveData))
        print(positiveData)

        with open('ExposedData_Healthy_TOTAL_test.txt', 'r') as filehandle:
            negativeData = json.load(filehandle)
        print(len(positiveData))
        print(negativeData)

    else:
        with open("ExposedData_Sick.txt", "r")as filehandle:
            positiveData = json.load(filehandle)

        print(positiveData)

        with open('ExposedData_Healthy.txt', 'r') as filehandle:
            negativeData = json.load(filehandle)
        print(negativeData)
        #######################
        with open('ExposedData_Sick_test.txt', 'r') as filehandle:
            positiveDataTest = json.load(filehandle)
        print(positiveDataTest)
        with open('ExposedData_Healthy_test.txt', 'r') as filehandle:
            negativeDataTest = json.load(filehandle)
        print(negativeDataTest)


# extractData("total")

training_epochs = 500
learning_rate = 0.0001
batch_size = 100
display_step = 1

n_input = 92
n_hidden_1 = 70
n_hidden_2 = 30
n_classes = 1

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
