
import tensorflow as tf

# Data
tf.app.flags.DEFINE_string('input_training_data','D:\\Projects\\Python\\CLoseVariants_LR\\TestData\\TrainingData_20180909.txt','training data path')

# Mode
tf.app.flags.DEFINE_string('mode','predict','train, predict or evaluation mode')
tf.app.flags.DEFINE_integer('feature_size', 1 << 29,'train, predict or evaluation mode')


# Paramters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("traing_epochs", 5, "Training epochs.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size.")
tf.app.flags.DEFINE_integer("buffer_size", 1, "Batch size.")
tf.app.flags.DEFINE_integer("dispaly_step", 100, "Display step.")

tf.app.flags.DEFINE_integer('loss_cnt',1,'total loss count to update')

tf.app.flags.DEFINE_string('output_model_path','model','path to save model')
tf.app.flags.DEFINE_string('log_dir','log_folder','folder to save log')
tf.app.flags.DEFINE_integer('log_frequency', 2, 'log frequency during training procedure')
tf.app.flags.DEFINE_integer('checkpoint_frequency', 2, 'evaluation frequency during training procedure')

tf.app.flags.DEFINE_integer('max_model_to_keep',10, 'max models to save')
tf.app.flags.DEFINE_string('result_filename','predict_res.txt','result file name')

FLAGS = tf.app.flags.FLAGS