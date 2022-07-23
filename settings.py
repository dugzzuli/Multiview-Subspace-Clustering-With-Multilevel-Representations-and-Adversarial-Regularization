import tensorflow.compat.v1 as tf


flags =tf.compat.v1.flags
FLAGS = flags.FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')

#不需要
flags.DEFINE_float('learning_rate_main', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('reg_ssc_param', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('cost_ssc_param', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('diver_param', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('IV_param', 64, 'Number of units in hidden layer 3.')