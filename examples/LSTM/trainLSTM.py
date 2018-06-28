import os
from optparse import OptionParser

# Command line options
parser = OptionParser()
parser.add_option("-l", "--loadModel", action="store_true", dest="loadModel", default=False, help="Load model")
parser.add_option("--useRMS", action="store_false", dest="useRMS", default=True, help="Use RMS")
parser.add_option("-f", "--filename", action="store", type="string", dest="filename", default="filenames.txt", help="File containing filenames of data files")
parser.add_option("-w", "--windowsize", action="store", type="int", dest="windowSize", default=30, help="Window size used for iterating the sequence (Samples)")
parser.add_option("-s", "--stridesize", action="store", type="int", dest="strideSize", default=1, help="Stride size used for iterating the sequence (Samples)")
parser.add_option("-u", "--testskip", action="store", type="int",  dest="testSkip", default=4, help="Reading to be put in to the test dataset")
parser.add_option("-t", "--trainDataStatsFile", action="store", type="string",  dest="trainDataStatsFile", default="../MotionClassifier/trainData_stats.txt", help="Train data statistics file")

# Parse command line options
(options, args) = parser.parse_args()

print options

# Import Activity Recognition data
import input_data_activity
dataset = input_data_activity.DataSet()
dataset.prepareData(options)

print "Data loaded"
print "Train data: ", dataset.trainData.shape
print "Train labels: ", dataset.trainTargetLabels.shape
print "Test data: ", dataset.testData.shape
print "Test labels: ", dataset.testTargetLabels.shape

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.platform import gfile
import freeze_graph
import numpy as np
import sys

# Parameters
learning_rate = 0.001
decay = 0.00001
training_iters = 30001
batch_size = 5000
test_len = len(dataset.testData)
display_step = 50
checkpoint_step = 1000
test_set_accuracy_step = 1000
checkpoint_dir = "/home/shoaib/Documents/EpiWear/Classifier/MotionClassifier/model/"
best_checkpoint_dir = checkpoint_dir + "best_model/"

graph_dir = "/home/shoaib/Documents/EpiWear/Classifier/MotionClassifier/graph/"
input_graph_name = "LSTM_MotionClassification.pb"
output_graph_name = "Graph_MotionClassification.pb"

checkpoint_prefix = os.path.join(best_checkpoint_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"

# Network Parameters
n_input = 3 # Data: 30 Values
n_steps = 10 # Feeding 3 values per timestep for 10 timesteps
n_hidden = 16 # hidden layer num of features
n_classes = 2 # Total acitivities

with tf.variable_scope('Motion_Classifier'):
    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input], name="Input_X")
    istate = tf.placeholder("float", [None, 2 * n_hidden], name="Input_State") #state & cell => 2x n_hidden
    y = tf.placeholder("float", [None, n_classes], name="Input_Y")

    # Define weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name="Hidden_Weights"), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name="Out_Weights")
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden]), name="Hidden_Biases"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="Out_Biases")
    }

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    with tf.variable_scope('Motion_Classifier'):
        outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    with tf.get_default_graph().name_scope('Motion_Classifier_Ops'):
        intermediate = tf.matmul(outputs[-1], _weights['out'])
        logits = tf.add(intermediate, _biases['out'])
        predictions = tf.nn.softmax(logits, name="output_node")
        return logits

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
testAccuracy = 0.0

test_data = dataset.testData.reshape((-1, n_steps, n_input))
test_label = dataset.testTargetLabels

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
bestModelSaver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

# Launch the graph
with tf.Session() as sess:
    if options.loadModel:
        print "Loading"

        # Restore Graph
        with gfile.FastGFile(graph_dir + output_graph_name,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            print "Graph Loaded"

        # persisted_result = sess.graph.get_tensor_by_name("Anomaly/Hidden_Weights:0")
        # tf.add_to_collection(tf.GraphKeys.VARIABLES, persisted_result)

        saver = tf.train.Saver( tf.all_variables())  # defaults to saving all variables - in this case w and b

        # Restore Model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model loaded Successfully!"
        else:
            print "Model not found"
            exit()
    else:
        sess.run(tf.initialize_all_variables())
        step = 1
        # Keep training until reach max iterations
        while step < training_iters:
            batch_xs, batch_ys = dataset.getNextBatch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * n_hidden))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * n_hidden))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * n_hidden))})
                print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

            #Save the model
            if step % checkpoint_step == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step = step)

            #Check the accuracy on test data
            if step % test_set_accuracy_step == 0:          
                # Report accuracy on test data
                testAcc = sess.run(accuracy, feed_dict={x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))})
                print "Testing Accuracy:", testAcc

                # If its the best accuracy achieved so far, save the model
                if testAcc >= testAccuracy:
                    testAccuracy = testAcc
                    print "Saving the best model"
                    # bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
                    bestModelSaver.save(sess, checkpoint_prefix, global_step=0,
                        latest_filename=checkpoint_state_name)
                else:
                    print "Previous best accuracy: ", testAccuracy
                
            step += 1

              # Write Graph to file
        print "Writing Graph to File"
        os.system("rm -rf " + graph_dir)
        tf.train.write_graph(sess.graph_def, graph_dir, input_graph_name, as_text=False) #proto

        # We save out the graph to disk, and then call the const conversion routine.
        input_graph_path = graph_dir + input_graph_name
        input_saver_def_path = ""
        input_binary = True
        input_checkpoint_path = checkpoint_prefix + "-0"
        output_node_names = "Motion_Classifier_Ops/output_node"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = graph_dir + output_graph_name
        clear_devices = False

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_path,
                                  clear_devices)

        print "Optimization Finished!"

    # Report accuracy on test data
    print "Testing Accuracy (currently):", sess.run(accuracy, feed_dict={x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))})

    # Report accuracy on test data using best fitted model
    ckpt = tf.train.get_checkpoint_state(best_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        bestModelSaver.restore(sess, ckpt.model_checkpoint_path)
    print "Testing Accuracy (using best model):", sess.run(accuracy, feed_dict={x: test_data, y: test_label, istate: np.zeros((test_len, 2 * n_hidden))})


    print "Testing the saved Graph"
    output_graph_path = graph_dir + output_graph_name
    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            output_node = sess.graph.get_tensor_by_name("Motion_Classifier_Ops/output_node:0")
            Input_X = sess.graph.get_tensor_by_name("Motion_Classifier/Input_X:0")
            # Input_Y = sess.graph.get_tensor_by_name("Anomaly_BreathingRate/Input_Y:0")
            Input_State = sess.graph.get_tensor_by_name("Motion_Classifier/Input_State:0")

            output = sess.run(output_node, feed_dict={Input_X: test_data, Input_State: np.zeros((test_len, 2 * n_hidden))})
            # print output
            print len(output), "==", len(test_data)
            assert(len(output) == len(test_data))

            print "Graph tested"