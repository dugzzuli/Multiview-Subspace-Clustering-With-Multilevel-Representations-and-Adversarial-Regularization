
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pickle

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
w_init = lambda:tf.random_normal_initializer(stddev=0.02)


class MLRGaussModel(object):

    def __init__(self, config):
        self.config = config
        self.att_shape = config['att_shape']
        self.batch_size = config['batch_size']

        self.num_att_layers = len(self.att_shape)

        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']

        if self.is_init:
            if os.path.isfile(self.pretrain_params_path):
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)


    def forward_att(self, x, drop_prob,view='1', reuse=False):


        with tf.variable_scope('V'+str(view)+'_encoder', reuse=reuse) as scope:
            self.att_input_dim = x.shape[1]
            cur_input = x
            print(cur_input.get_shape())
            cur_inpu_List=[]
            cur_inpu_List.append(cur_input) #为输出层也添加一层自表征
            Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_C') #创建公共表征
            Coef=Coef - tf.diag(tf.diag_part(Coef))

            D_list=[]
            tempD=tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_D0')
            tempD=tempD-tf.diag(tf.diag_part(tempD))
            D_list.append(tempD) #为输出层添加多样性表征

            # ============encoder===========
            struct = self.att_shape
            for i in range(self.num_att_layers):
                name = 'V'+view+'_encoder' + str(i)
                if self.is_init:
                    print('V' + view + '_encoder:=====cur_input=====:' + str(i))
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_att_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                cur_inpu_List.append(cur_input)

                tempD =tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_D{}'.format(i+1))
                tempD = tempD - tf.diag(tf.diag_part(tempD))
                D_list.append(tempD)  # 为输出层添加多样性表征

                print(cur_input.get_shape())

            Y_list=[]
            for cur,D in zip(cur_inpu_List,D_list):
                D=D - tf.diag(tf.diag_part(D))
                tempY=tf.matmul(Coef+D, cur)
                Y_list.append(tempY)
            # ====================decoder=============
        with tf.variable_scope('V' + str(view) + '_decoder', reuse=reuse) as scope:
            struct.reverse()
            Y_list.reverse()
            cur_input = Y_list[0]
            Y=tf.zeros_like(cur_input)
            for i in range(self.num_att_layers - 1):
                name = 'V'+view+'_decoder' + str(i)

                if self.is_init:
                    print('V' + view + '_decoder:=====cur_input=====:' + str(i))
                    cur_input = tf.layers.dense(cur_input+Y, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input+Y, units=struct[i + 1], kernel_initializer=w_init())
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())
                Y=Y_list[i+1]

            name = 'V'+view+'_decoder' + str(self.num_att_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input+Y_list[-2], units=self.att_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input+Y_list[-2], units=self.att_input_dim, kernel_initializer=w_init())
            # cur_input = tf.nn.sigmoid(cur_input)

            cur_input=cur_input+Y_list[-1]
            # cur_input=cur_input

            x_recon = cur_input
            print(cur_input.get_shape())

            self.att_shape.reverse()

            Y_list.reverse()

        print("=====forward_V" + view + "------end=====")
        return cur_inpu_List,Y_list, x_recon,Coef,D_list



class MVGaussModel(object):
    def __init__(self,config):
        self.config = config
        self.mvList=[]

        self.View_num = config['View_num']

    def getModel(self):
        for i in range(self.View_num):
            self.mvList.append(MLRGaussModel(self.config))
        return  self.mvList


