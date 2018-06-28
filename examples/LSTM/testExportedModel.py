import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import sys

graphFileMotion = "/home/shoaib/Documents/EpiWear/Classifier/MotionClassifier/graph/Graph_MotionClassification.pb"
trainDataStatsFileMotion = "/home/shoaib/Documents/EpiWear/Classifier/MotionClassifier/trainData_stats.txt"

# Network Parameters
n_input = 3 # Data: 30 Values
n_steps = 10 # Feeding 3 values per timestep for 10 timesteps
n_hidden = 16 # hidden layer num of features


class MotionClassificationLSTM:

	def loadModel(self):
		with tf.Graph().as_default():
			output_graph_def = tf.GraphDef()

			# Load the normalization params
			f = open(trainDataStatsFileMotion, "r+")
			self.trainData_min = float(f.readline().strip())
			self.trainData_max = float(f.readline().strip())
			f.close()

			# Load graph
			with open(graphFileMotion, "rb") as f:
				output_graph_def.ParseFromString(f.read())
				_ = tf.import_graph_def(output_graph_def, name="")

				self.session = tf.Session()
				self.output_node = self.session.graph.get_tensor_by_name("Motion_Classifier_Ops/output_node:0")
				self.Input_X = self.session.graph.get_tensor_by_name("Motion_Classifier/Input_X:0")
				# Input_Y = sess.graph.get_tensor_by_name("Motion_Classifier/Input_Y:0")
				self.Input_State = self.session.graph.get_tensor_by_name("Motion_Classifier/Input_State:0")



	def classify(self, data):
		sequence = (data - self.trainData_min) / float(self.trainData_max - self.trainData_min)
		sequence = sequence.reshape((-1, n_steps, n_input))
		
		# Make predictions
		prediction = self.session.run(self.output_node, feed_dict={self.Input_X: sequence, self.Input_State: np.zeros((1, 2 * n_hidden))})
		return prediction[0][1]


	def deleteSession(self):
		self.session.close()



if __name__ == "__main__":
	motionClassificationLSTM = MotionClassificationLSTM()
	motionClassificationLSTM.loadModel()
	data = np.loadtxt('seizureVecs.txt')
	correct = 0
	total = 0
	for row in data:
		seizureMovementConfidenceScore = motionClassificationLSTM.classify(row)
		if seizureMovementConfidenceScore >= 0.5:
			correct += 1

		total += 1

	print "Accuracy:", float(correct) / total 