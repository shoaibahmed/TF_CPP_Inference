import tensorflow as tf
import numpy as np
import cv2

import datasets.fsns as fsns

# Load the saved graph definition
checkpoint_dir = "./checkpoint/"
output_graph = checkpoint_dir + "frozen_model.pb"

charset = fsns.read_charset("./datasets/testdata/fsns/charset_size=134.txt")
# print (charset)

with tf.Graph().as_default():
	output_graph_def = tf.GraphDef()

	with open(output_graph, "rb") as f:
		output_graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(output_graph_def, name="")

	with tf.Session() as session:
		imagePlaceholder = session.graph.get_tensor_by_name("imageBatchPlaceholder:0")
		predictedCharsNode = session.graph.get_tensor_by_name("AttentionOcr_v1/predicted_chars:0")

		# Feed an input image
		# image = cv2.imread("./test/body_shop.jpg")
		image = cv2.imread("./test/avenue.png")
		resized_image = cv2.resize(image, (600, 150)) 
		batchIm = np.expand_dims(resized_image, 0)
		predictedChars = session.run(predictedCharsNode, feed_dict={imagePlaceholder: batchIm})

		# Iterate over the images in batch
		for predictedImageChars in predictedChars:
			print ("Predicted characters: %s" % predictedImageChars)

			finalString = ''
			for c in predictedImageChars:
				finalString += charset[c]

			# finalString = unicode(finalString, "utf-8")
			print ("Converted string: %s" % finalString.encode('utf-8'))