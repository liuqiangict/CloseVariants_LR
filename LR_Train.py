
import os
import sys
import time

import tensorflow as tf
import numpy as np

from params import FLAGS
from DataReader import DataReader
from Trainer import Trainer
from LR import LRModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def Train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        inc_step = tf.assign_add(global_step, 1)

        reader = DataReader(FLAGS.input_training_data, FLAGS.buffer_size, FLAGS.batch_size, FLAGS.traing_epochs, is_shuffle=True)
        model = LRModel()
        trainer = Trainer(model, inc_step, reader)

        summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True

        saver = tf.train.Saver(max_to_keep = FLAGS.max_model_to_keep, name = 'model_saver')

        with tf.Session(config = config) as session:
            summ_writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
            #Load Pretrain
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            session.run(reader.iterator.initializer)

            #zero = tf.constant(0, dtype=tf.float32)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.output_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print("Load model from ", ckpt.model_checkpoint_path)
            else:
                print("No initial model found.")

            trainer.start_time = time.time()
            while True:
                try:
                    _, avg_loss, total_weight, step, summary = session.run(trainer.train_ops() + [summary_op])
                   
                    #where = tf.not_equal(model.W, zero)
                    #indices = tf.where(where)
                    #print(session.run(indices))
                    #print(session.run(tf.gather(model.W, indices)))
                    #print(session.run(model.b))
                    #print(step)
                    if step % FLAGS.log_frequency == 1:
                        summ_writer.add_summary(summary,step)
                        trainer.print_log(total_weight, step, avg_loss)
                    if step % FLAGS.checkpoint_frequency == 1:
                        saver.save(session, FLAGS.output_model_path + "/model", global_step=step)
                except tf.errors.OutOfRangeError:
                    print("End of training.")
                    break
    pass


def Predict():
    outputter = tf.gfile.GFile(FLAGS.output_model_path + "/" + FLAGS.result_filename, mode = "w")
    mode = tf.contrib.learn.ModeKeys.INFER
    reader = DataReader('predict.txt', FLAGS.buffer_size, FLAGS.batch_size, FLAGS.traing_epochs, is_shuffle=False)
    model = LRModel()
    trainer = Trainer(model, None, reader)

    scope = tf.get_variable_scope()
    scope.reuse_variables()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True

    with tf.Session(config = config) as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        session.run(reader.iterator.initializer)

        ckpt = tf.train.get_checkpoint_state(FLAGS.output_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print("Load model from ", ckpt.model_checkpoint_path)
        else:
            print("No initial model found.")

        #zero = tf.constant(0, dtype=tf.float32)
        #where = tf.not_equal(model.W, zero)
        #indices = tf.where(where)
        #print(session.run(indices))
        #print(session.run(tf.gather(model.W, indices)))
        #print(session.run(model.b))
        trainer.predict(session, mode, outputter)
    outputter.close()
    pass

if __name__ == '__main__':
    #Create folders
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.output_model_path):
        tf.gfile.MakeDirs(FLAGS.output_model_path)
    if FLAGS.mode == 'train':
        Train()
    elif FLAGS.mode == 'predict' or FLAGS.mode == 'eval':
        Predict()