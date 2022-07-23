import os

from Utils import *
from SingleAE import SingleAE
import pickle


class PreTrainer(object):

    def __init__(self, config, learning_rate=1e-3,epoches=1):
        self.epoches=epoches
        self.learning_rate = learning_rate
        self.config = config
        self.att_input_dim = -1
        self.View = config['View']
        self.batch_size = config['batch_size']


        self.pretrain_params_path = config['pretrain_params_path']

        mkdir(os.path.dirname(os.path.abspath(self.pretrain_params_path)))

        self.W_init = {}
        self.b_init = {}

    def pretrain(self, data, modal):
        self.att_input_dim = data.shape[1]

        shape = [self.att_input_dim] + self.View

        for i in range(len(shape) - 1):
            print(shape[i], shape[i + 1])

            activation_fun1 = lrelu
            activation_fun2 = lrelu
            if i == 0:
                activation_fun2 = None
            if i == len(shape) - 2:
                activation_fun1 = None

            SAE = SingleAE([shape[i], shape[i + 1]],
                           {"iters": self.epoches, "batch_size": 256, "lr": self.learning_rate, "dropout": 0.8}, data,
                           i, activation_fun1, activation_fun2)
            SAE.doTrain()
            W1, b1, W2, b2 = SAE.getWb()

            name = modal + "_encoder" + str(i)
            self.W_init[name] = W1
            self.b_init[name] = b1
            name = modal + "_decoder" + str(len(shape) - i - 2)
            self.W_init[name] = W2
            self.b_init[name] = b2

            data = SAE.getH()


        mkdir(self.pretrain_params_path.replace('pretrain_params.pkl',''))
        with open(self.pretrain_params_path, 'wb') as handle:
            pickle.dump([self.W_init, self.b_init], handle, protocol=pickle.HIGHEST_PROTOCOL)


