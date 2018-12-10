
import sys
import tensorflow as tf
from GenerateCrossFeatures import GenerateCrossFeatures

class DataReader():
    def __init__(self, filename, buffer_size, batch_size, num_epochs, is_shuffle=True):
        self.crossFeaturer = GenerateCrossFeatures()
        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.map(self.parse_line, num_parallel_calls = 1)
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size = buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset_batch = dataset.batch(batch_size)
        dataset_batch = dataset_batch.prefetch(buffer_size=buffer_size)
        self.iterator = dataset_batch.make_initializable_iterator()

    def parse_line(self, line):
        columns = tf.decode_csv(line, [[""] for i in range(0, 3)], field_delim="\t", use_quote_delim=False)
        res = self.convert_to_xletter(columns)
        return res

    def convert_to_xletter(self, columns):
        res = tf.py_func(self.crossFeaturer.GetFeature, [columns[0], columns[1]], [tf.string])
        res = [columns[0], columns[1]] + res +  [tf.string_to_number(columns[2], tf.float32)]
        return res

    def get_next(self):
        return self.iterator.get_next()

if __name__ == '__main__':
    reader = DataReader('predict.txt', 1, 1, 1, False)
    with tf.Session() as sess:
        sess.run(reader.iterator.initializer)
        for i in range(0,9):
            #print(i)
            x, y = reader.get_next()
            #print(sess.run(x))
            print(sess.run(y))
            #features = tf.string_split(x, ';')
            #print(sess.run(features))
            #features_tensor = tf.SparseTensor(indices=features.indices, values=tf.string_to_number(features.values, out_type=tf.int32), dense_shape=features.dense_shape)
            
            #print(sess.run(features_tensor))
            #print(sess.run(y))
