
import sys
import tensorflow as tf

from params import FLAGS


class LRModel():
    def __init__(self):
        self.W = tf.get_variable(name='Weights', shape=[FLAGS.feature_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        self.b = tf.get_variable(name='Bisas', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0)) 
        pass

    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            str_features, str_labels = input_fields[2], input_fields[3]
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            str_features, str_labels = input_fields[2], "1"
        features = tf.string_split(str_features, ';')
        features = tf.SparseTensor(
            indices = features.indices,
            values=tf.string_to_number(features.values, out_type=tf.int32),
            dense_shape=features.dense_shape)
        labels = tf.reshape(str_labels, [-1])
        
        product = tf.nn.embedding_lookup_sparse(self.W, features, None, combiner='sum')
        pred = tf.nn.sigmoid(product + self.b)

        return input_fields[0], input_fields[1], pred, labels

    def calc_loss(self, inference_res):
        a, b, pred, labels = inference_res
        batch_size = tf.shape(pred)[0]
        
        cost = tf.losses.log_loss(labels=labels, predictions=pred)
        loss = tf.reduce_sum(cost)
        weight = batch_size

        return [loss], weight

    def predict(self, inference_res):
        a, b, pred, labels = inference_res
        return a, b, pred

    def get_optimizer(self, optimizer_mode = "Grad"):
        if optimizer_mode == "Adam":
            return [tf.train.AdadeltaOptimizer(FLAGS.learning_rate)]
        elif optimizer_mode == "Grad":
            return [tf.train.GradientDescentOptimizer(FLAGS.learning_rate)]
        elif optimizer_mode == "FTRL":
            return [tf.train.FtrlOptimizer(FLAGS.learning_rate)]