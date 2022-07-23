import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ModelLossFromARGA import Discriminator,DiscriminatorUniversal
from Utils import mkdir

flags = tf.app.flags
FLAGS = flags.FLAGS
import numpy as np




import time

time_start = time.time()

class TrainerMLRMVGaussUniversalDistribution():
    '''
    高斯对抗分布

    '''
    def __init__(self, model, config, loadW=True):

        self.model = model.getModel()


        self.zeroconstant = tf.constant(0, dtype=tf.float32)


        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.diver_param = self.config['diver_param']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.reg_ssc_param_2 = self.config['reg_ssc_param_2']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']
        self.IV_param = self.config['IV_param']
        self.lambdaL_param = self.config['lambdaL_param']
        self.loadW = loadW

        self.dis=DiscriminatorUniversal(V=0)



        self.iter = 0

        self.xList = []
        for i in range(self.View_num):
            self.xList.append(tf.placeholder(tf.float32, [None, self.dims[i]], name='V' + str(i + 1)))
        if (self.loadW):
            self.adjs = []
            for i in range(self.View_num):
                self.adjs.append(tf.placeholder(tf.float32, [None, None], name='adj' + str(i + 1)))

        # self.x = tf.placeholder(tf.float32)

        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.Q = tf.placeholder(tf.float32, [None, None], name="Q")

        self.optimizer,self.optimizer_ssc,self.optimizer_ssc_disc_var, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.diver_cost, self.layer_adj_loss, self.IV_cost, \
        self.Q_loss, self.Coef, self.D_list,self.genLoss,self.disLoss = self._build_training_graph()




    def initlization(self):
        #
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

    def get_1st_loss(self, H, adj_mini_batch):
        D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
        L = D - adj_mini_batch  ## L is laplation-matriX
        return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

    def HSIC(self, c_v, c_w):
        N = tf.shape(c_v)[0]
        H = tf.ones((N, N)) * tf.cast((1 / N), tf.float32) * (-1) + tf.eye(N)
        K_1 = tf.matmul(c_v, tf.transpose(c_v))
        K_2 = tf.matmul(c_w, tf.transpose(c_w))
        rst = tf.matmul(K_1, H)
        rst = tf.matmul(rst, K_2)
        rst = tf.matmul(rst, H)
        rst = tf.trace(rst)
        return rst

    def _build_training_graph(self):
        """

        :return:
        """

        cur_inpu_List_All_view = []
        Y_list_All_view = []
        x_recon_All_view = []
        Coef_All_view = []
        D_list_All_view = []
        for view in range(self.View_num):
            cur_inpu_List_view, Y_list_view, x_recon_view, Coef_view, D_list_view = self.model[view].forward_att(
                self.xList[view], drop_prob=self.drop_prob, view=str(view + 1), reuse=False)

            cur_inpu_List_All_view.append(cur_inpu_List_view)
            Y_list_All_view.append(Y_list_view)
            x_recon_All_view.append(x_recon_view)
            Coef_All_view.append(Coef_view)
            D_list_All_view.append(D_list_view)

        # #重构正则
        recon_ssc = self.zeroconstant
        # recon_ssc=recon_ssc+tf.reduce_sum(tf.pow(tf.subtract(x_recon, self.x), 2.0))
        # #子空间表征正则
        #
        cost_ssc = 0
        for cur_inpu_List, Y_list in zip(cur_inpu_List_All_view, Y_list_All_view):

            assert len(self.cost_ssc_param) == len(Y_list), '子空间和参数长度不等'

            for cur, Y, p in zip(cur_inpu_List, Y_list, self.cost_ssc_param):
                cost_ssc = cost_ssc + p * tf.reduce_sum(tf.pow(tf.subtract(cur, Y), 2.0))

        #  -tf.reduce_mean(Y * tf.log(tf.clip_by_value(cur, 1e-10, 1.0)))

        layer_adj_loss = tf.Variable(0, name='layer_adj_loss', dtype=tf.float32)

        # self.lambdaL=[10,40,100,40,10]
        if (self.loadW):
            for layer_o in cur_inpu_List_All_view:

                assert len(layer_o) == len(self.lambdaL_param), '参数不相等'

                for i, o, L in zip(range(self.View_num), layer_o, self.lambdaL_param):
                    if i == 0 or i == self.View_num - 1:
                        continue
                    layer_adj_loss += L * self.get_1st_loss(o, self.adjs[i])

        IV_cost = 0
        for Coef in Coef_All_view:
            for Coef_j in Coef_All_view:
                if (Coef != Coef_j):
                    # IV_cost = IV_cost + self.HSIC(Coef_j, Coef)
                    IV_cost = IV_cost + tf.reduce_sum(tf.multiply(Coef_j, Coef))
                    # IV_cost = IV_cost + tf.norm(tf.multiply(Coef_j, Coef),1)

        diver_cost = 0
        for D_list in D_list_All_view:
            for D in D_list:
                for DJ in D_list:
                    if (D != DJ):
                        # diver_cost = diver_cost + tf.trace(tf.matmul(tf.transpose(D), DJ))
                        diver_cost = diver_cost + tf.reduce_sum(tf.multiply(D, DJ))
                    # self.HSIC(D,DJ)

        Qloss = 0
        # # ord=1
        # # tempQ=tf.matmul(tf.transpose(self.Q), tf.abs(Coef))
        # # Qloss=tf.norm(tempQ, ord)
        #
        # #Self-Supervised Convolutional Subspace Clustering Network

        for Coef in Coef_All_view:
            A = tf.add(tf.transpose(Coef), Coef) / 2
            D = tf.diag(tf.reduce_sum(A, 1))

            Qloss = Qloss + 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), D - A), self.Q))

        for D_list in D_list_All_view:
            for D in D_list:
                AD = tf.add(tf.transpose(D), D) / 2
                DD = tf.diag(tf.reduce_sum(AD, 1))
                Qloss = Qloss + 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), DD - AD), self.Q))

        #
        #
        # #亲和矩阵正则
        reg_ssc_cost = 0
        for Coef in Coef_All_view:
            reg_ssc_cost = reg_ssc_cost + self.reg_ssc_param*tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + self.reg_ssc_param_2*tf.norm(D, 2) ** 2


            #
        loss_ssc = cost_ssc + reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

        d_fake_list=[]
        d_real_list=[]

        for cur in cur_inpu_List_All_view:
            curInput = cur[-1]
            d_fake=self.dis.construct(curInput)

            prior = tf.random_uniform(shape=tf.shape(curInput))

            d_real=self.dis.construct(prior)

            d_fake_list.append(d_fake)
            d_real_list.append(d_real)

        disLoss = 0
        genLoss=0
        for d_real,d_fake in zip(d_real_list,d_fake_list):
            # Discrimminator Loss
            dc_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real,name='dclreal'))

            dc_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
            dc_loss = dc_loss_fake + dc_loss_real

            disLoss = disLoss + dc_loss

            # Generator loss
            generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))
            genLoss=genLoss+generator_loss

        # genLoss=genLoss+loss_ssc

        all_variables = tf.trainable_variables()

        gen_var = [var for var in all_variables if 'encoder' in var.name]

        disc_var = [var for var in all_variables if 'Discriminator' in var.name]

        ae_var = [var for var in all_variables if "encoder" or "decoder" in var.name]

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(genLoss,var_list=gen_var)

        optimizer_ssc_disc_var = tf.train.AdamOptimizer(learning_rate=self.learning_rate/5).minimize(disLoss, var_list=disc_var)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc,var_list=ae_var)  # Adam Optimizer

        return optimizer,optimizer_ssc,optimizer_ssc_disc_var, loss_ssc, cost_ssc, recon_ssc, reg_ssc_cost, diver_cost, layer_adj_loss, IV_cost, Qloss, Coef_All_view, D_list_All_view,genLoss,disLoss

    def finetune_fit(self, graph, learning_rate, Q):
        XList = graph.ViewData
        hasW = hasattr(graph, 'W')

        if hasW:
            WList = graph.W

        XList_feed = {}
        for i in range(self.View_num):
            XList_feed["V" + str(i + 1) + ":0"] = XList[i]
            if hasW:
                XList_feed["adj" + str(i + 1) + ":0"] = WList[i]

        XList_feed["Q:0"] = Q
        XList_feed["learning_rate:0"] = learning_rate



        for j in range(1):
            _= self.sess.run([self.optimizer], feed_dict=XList_feed)

        for j in range(1):
            C, D_list, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,disLoss = self.sess.run(
                (self.Coef, self.D_list, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
                 self.diver_cost, self.layer_adj_loss, self.IV_cost, \
                 self.Q_loss,self.disLoss), \
                feed_dict=XList_feed)

        for j in range(2):
            _,genLoss=self.sess.run([self.optimizer_ssc_disc_var,self.genLoss], feed_dict=XList_feed)

        self.iter = self.iter + 1

        t = time.time()-time_start
        print("tiime:{}\n".format(t))
        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")
