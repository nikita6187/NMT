import tensorflow as tf
import numpy as np
import time
import sys


embedding_size = int(sys.argv[2])
vocabulary_size = int(sys.argv[3])


I = tf.placeholder(tf.int32, shape=())
B = tf.cast(I / 10, tf.int32)
f = embedding_size


word_embeddings = tf.get_variable("word_embeddings", [1, 1, embedding_size, vocabulary_size])

# Generate random decoder output
decoder_output = tf.random_uniform(shape=[I, B, f])

# Get distance
#word_embeddings_dis = tf.expand_dims(tf.expand_dims(tf.transpose(word_embeddings, perm=[1, 0]), axis=0), axis=0)  # [1, 1, f=embedding_size, vocab]

decoder_output_dis = tf.expand_dims(decoder_output, axis=-1)  # [I, B, f, 1]
#distances = tf.assign_sub(word_embeddings, decoder_output)  # [I, B, f, vocab]

word_embeddings = word_embeddings - decoder_output_dis  # [I, B, f, vocab]
distances = word_embeddings
distances = tf.pow(distances, 2)  # [I, B, f, vocab]
distances = tf.reduce_sum(distances, axis=2)  # [I, B, vocab]
normalization_constant = 1 / tf.reduce_sum(distances, axis=-1, keepdims=True)  # [I, B, 1]
output_geometric = tf.multiply(distances, normalization_constant)  # [I, B, vocab]

output_softmax = tf.layers.dense(decoder_output, units=vocabulary_size, activation=tf.nn.softmax)

#c = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
c = None

with tf.Session(config=c) as sess:
    sess.run(tf.initializers.global_variables())

    print("\n\n\n\n")

    times_geo = []
    for i in range(500):
        time_a = time.time()
        sess.run(output_geometric, feed_dict={I: int(sys.argv[1])})
        times_geo.append(time.time() - time_a)

    times_soft = []
    for i in range(500):
        time_a = time.time()
        sess.run(output_softmax, feed_dict={I: int(sys.argv[1])})
        times_soft.append(time.time() - time_a)

    print("\n\n")
    print(times_geo[-20:])
    print(times_soft[-20:])
    print("Time geometric: " + str(sum(times_geo)/float(len(times_geo))) + " time softmax: " + str(sum(times_soft)/float(len(times_soft))))





