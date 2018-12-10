
import os
import sys
import time

import numpy as np
import tensorflow as tf

from datetime import datetime
from params import FLAGS


class Trainer:
    def __init__(self, model, inc_step, train_reader, eval_reader = None, infer_reader = None):
        self.model = model

        self.inc_step = inc_step
        self.reader = train_reader
        self.eval_reader = eval_reader
        self.infer_reader = infer_reader

        self.devices = self.get_devices()
        self.total_weight = [tf.Variable(0., trainable=False) for i in range(0, FLAGS.loss_cnt)]
        self.total_loss = [tf.Variable(0., trainable=False) for i in range(0, FLAGS.loss_cnt)]
                
        opts = self.model.get_optimizer()
        tower_grads = []
        tower_loss = [[] for i in range(0,FLAGS.loss_cnt)]

        self.weight_record = 0

        if FLAGS.mode == 'train':
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    with tf.name_scope('Device_%d' % i) as scope:
                        batch_input = self.reader.get_next()
                        loss, weight = self.tower_loss(batch_input)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = []
                        for opt in opts:
                            grads.extend(opt.compute_gradients(loss[0]))
                        tower_grads.append(grads)
                        for j in range(0, len(loss)):
                            tower_loss[j].append((loss[j], weight))
            self.avg_loss = [self.update_loss(tower_loss[i], i) for i in range(0, len(tower_loss))]
            grads = self.sum_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads)

        if FLAGS.mode == 'predict':
            tower_infer = []
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    infer_batch = self.reader.get_next()
                    infer_res = self.tower_inference(infer_batch)
                    tower_infer.append([infer_batch, infer_res])
            self.infer_list = self.merge_infer_res(tower_infer)
        pass

    def tower_loss(self, batch_input):
        inference_output = self.model.inference(batch_input,tf.contrib.learn.ModeKeys.TRAIN)
        loss,weight = self.model.calc_loss(inference_output)
        tf.summary.scalar("losses",loss[0])
        return loss, weight

    def update_loss(self, tower_loss, idx):
        loss, weight = zip(*tower_loss)
        loss_inc = tf.assign_add(self.total_loss[idx], tf.reduce_sum(loss))
        weight_inc = tf.assign_add(self.total_weight[idx], tf.cast(tf.reduce_sum(weight), tf.float32))
        avg_loss = loss_inc / weight_inc
        tf.summary.scalar("avg_loss" + str(idx), avg_loss)
        return avg_loss

    def sum_gradients(self, tower_grads):
       sum_grads = []
       print(tower_grads)
       for grad_and_vars in zip(*tower_grads):
           print(grad_and_vars)
           if isinstance(grad_and_vars[0][0],tf.Tensor):
               print(grad_and_vars[0][0])
               grads = []
               for g, _ in grad_and_vars:
                   expanded_g = tf.expand_dims(g,0)
                   grads.append(expanded_g)
               print(grads)
               grad = tf.concat(grads, 0)
               grad = tf.reduce_sum(grad, 0)
               v = grad_and_vars[0][1]
               grad_and_var = (grad, v)
               sum_grads.append(grad_and_var)
           else:
               values = tf.concat([g.values for g,_ in grad_and_vars],0)
               indices = tf.concat([g.indices for g,_ in grad_and_vars],0)
               v = grad_and_vars[0][1]
               grad_and_var = (tf.IndexedSlices(values, indices),v)
               sum_grads.append(grad_and_var)
       return sum_grads

    def get_devices(self):
        devices = []
        if os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
            for i, gpu_id in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
                gpu_id = int(gpu_id)
                if gpu_id < 0:
                    continue
                devices.append('/gpu:'+str(gpu_id))
        if not len(devices):
            devices.append('/cpu:0')
        print("available devices", devices)
        return devices

    
    def train_ops(self):
        return [self.train_op, self.avg_loss, self.total_weight, self.inc_step]

    def print_log(self, total_weight, step, avg_loss):
        examples, self.weight_record = total_weight[0] - self.weight_record, total_weight[0]
        current_time = time.time()
        duration, self.start_time = current_time - self.start_time, time.time()
        examples_per_sec = examples * 10000 / duration
        sec_per_steps = float(duration / FLAGS.log_frequency)
        format_str = "%s: step %d, %5.1f examples/sec, %.3f sec/step, %.1f samples processed,"
        avgloss_str = "avg_loss = " + ",".join([str(avg_loss[i]) for i in range(0,len(avg_loss))])
        print(format_str % (datetime.now(), step, examples_per_sec, sec_per_steps, total_weight[0]) + avgloss_str)
        pass

    def tower_inference(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.INFER)
        a, b, prediction = self.model.predict(inference_output)
        return a, b, prediction

    def merge_infer_res(self, tower_infer):
        infer_batch, infer_res = zip(*tower_infer)
        merge_batch = []
        merge_res = []
        for i in zip(*infer_batch):
            if not isinstance(i[0],tf.Tensor):
                merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
            else:
                merge_batch.append(tf.concat(i, axis=0))
        #for i in zip(*infer_res):
        #merge_res.append(tf.concat(infer_res[0][0], axis=0))
        #merge_res.append(tf.concat(infer_res[0][1], axis=0))
        merge_res.append(tf.concat(infer_res[0][2], axis=0))
        return merge_batch, merge_res

    def predict(self, sess, mode, outputter):
        assert(mode == tf.contrib.learn.ModeKeys.INFER)
        while True:
            try:
                #input_batch1, input_batch2, score = sess.run(self.scoring_ops())
                input_batch, score = sess.run(self.scoring_ops())
                for i in range(len(score)):
                    #output_str = ""
                    output_str = input_batch[0][i].decode("utf-8") + "\t" + input_batch[1][i].decode("utf-8") + "\t" + input_batch[2][i].decode("utf-8")  + "\t"
                    output_str += str(score[0][i])
                    outputter.write(output_str + "\n")
            except tf.errors.OutOfRangeError:
                print("score predict done.")
                break

    def scoring_ops(self):
        return self.infer_list