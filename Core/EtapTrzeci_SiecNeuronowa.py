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


postTestHard_Feature, postTestHard_Labels = fetchData("ExposedData_Sick_TOTAL_HARD_test.txt")
postTestLight_Feature, postTestLight_Labels = fetchData("ExposedData_Sick_TOTAL_SOFT_test.txt")

postTestNeg_Feature, postTestNeg_Labels = fetchData("ExposedData_Healthy_TOTAL_test.txt")

x_data_test = np.concatenate((postTestHard_Feature, postTestNeg_Feature, postTestLight_Feature), axis=0)
y_data_test = np.concatenate((postTestHard_Labels, postTestNeg_Labels, postTestLight_Labels), axis=0)
trainer = []
for x in y_data:
    if (x[0] == 0):
        trainer.append([0, 1])
    else:
        # x = [1, 0]
        trainer.append([1, 0])

trainer = np.array(trainer)

# print (trainer)
y_data = trainer

trainer = []
for y in y_data_test:
    if (y[0] == 0):
        trainer.append([0, 1])
    else:
        # x = [1, 0]
        trainer.append([1, 0])
trainer = np.array(trainer)

# print (trainer)
y_data_test = trainer

print(x_data.shape)
print(y_data_test)

# x_data = tf.convert_to_tensor(x_data, np.float32)
# y_data = tf.convert_to_tensor(y_data, np.float32)
# x_data_test = tf.convert_to_tensor(x_data_test, np.float32)
# y_data_test = tf.convert_to_tensor(y_data_test, np.float32)

'''





'''
# n_nodes_hl1 = 70
# n_nodes_hl2 = 50
# n_nodes_hl3 = 30

# n_nodes_hl1 = 500
# n_nodes_hl2 = 400
# n_nodes_hl3 = 300
# n_nodes_hl4 = 100
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 400
n_nodes_hl3 = 300
n_nodes_hl4 = 100
Epoch: 10000 cost= 0.174207214
Optimization Finished!
Accuracy: 0.7754237

'''

'''
n_nodes_hl1 = 500
n_nodes_hl2 = 400
n_nodes_hl3 = 300
n_nodes_hl4 = 100
Epoch: 10000 cost= 0.174207214
Optimization Finished!
Accuracy: 0.7245763

 z sigmoidami wszędzie 
'''
'''
n_nodes_hl1 = 700
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 150
Optimization Finished!
Accuracy: 0.7372881
'''

'''
n_nodes_hl1 = 150
n_nodes_hl2 = 100
n_nodes_hl3 = 92
n_nodes_hl4 = 50
Optimization Finished!
Accuracy: 0.78571427

# n_nodes_hl1 = 500
# n_nodes_hl2 = 400
# n_nodes_hl3 = 300
TEŻ BYŁO SPOKOOO
'''
n_nodes_hl1 = 150
n_nodes_hl2 = 100
n_nodes_hl3 = 92
n_nodes_hl4 = 50
# n_classes = 1

batch_size = 92
hm_epochs = 10000

n_input = x_data.shape[1]
n_classes = y_data.shape[1]

x = tf.placeholder('float', [None, 92])
y = tf.placeholder('float', [None, n_classes])


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([92, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.sigmoid(l4)

    output = tf.matmul(l4, output_layer['weights'] + output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
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

            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: x_data_test, y: y_data_test}))


# how to save model

train_neural_network(x)

