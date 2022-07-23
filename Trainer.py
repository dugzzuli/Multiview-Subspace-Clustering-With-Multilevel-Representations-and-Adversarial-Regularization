import tensorflow as tf

from ModelLossFromARGA import Discriminator,DiscriminatorUniversal
from Utils import mkdir
flags = tf.app.flags
FLAGS = flags.FLAGS
import numpy as np

class Trainer():
    def __init__(self, model, config):
        self.config = config
        self.model = model

        self.n_input = self.config["n_input"]
        self.batch_size = self.config["batch_size"]

        self.drop_prob = self.config['drop_prob']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']

        self.iter = 0
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.Q = tf.placeholder(tf.float32, [None, None])
        self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.Q_loss, self.Coef = self._build_training_graph()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)

    def _build_training_graph(self):
        att_H, att_H_SSC, x_recon, Coef = self.model.forward_att(self.x, drop_prob=self.drop_prob, reuse=False)

        # 子空间表征正则
        cost_ssc = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(att_H, att_H_SSC), 2))

        # 重构正则
        recon_ssc = tf.reduce_sum(tf.pow(tf.subtract(x_recon, self.x), 2.0))

        # Q  Multi-Level Representation Learning for Deep Subspace Clustering 有一点点效果但是好像不太高
        # ord=1
        # tempQ=tf.matmul(tf.transpose(self.Q), tf.abs(Coef))
        # Qloss=tf.norm(tempQ, ord)

        # Self-Supervised Convolutional Subspace Clustering Network
        A = tf.add(tf.transpose(Coef), Coef) / 2

        D = tf.diag(tf.reduce_sum(A, 1))
        Qloss = 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), D - A), self.Q))

        # 亲和矩阵正则
        reg_ssc = tf.reduce_sum(tf.pow(Coef, 2))

        loss_ssc = cost_ssc * self.cost_ssc_param + self.reg_ssc_param * reg_ssc + recon_ssc + Qloss * self.Q_param

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc)
        return optimizer_ssc, loss_ssc, cost_ssc, recon_ssc, reg_ssc, Qloss, Coef

    def finetune_fit(self, X, learning_rate, Q):
        C, reg_ssc, cost_ssc, _, Q_loss = self.sess.run(
            (self.Coef, self.reg_ssc, self.cost_ssc, self.optimizer_ssc, self.Q_loss), \
            feed_dict={self.x: X, self.learning_rate: learning_rate, self.Q: Q})
        self.iter = self.iter + 1
        return C, reg_ssc, cost_ssc, Q_loss

    def infer_clustering(self, X, learning_rate, Q):
        C, reg_ssc, cost_ssc, Q_loss = self.sess.run(
            (self.Coef, self.reg_ssc, self.cost_ssc, self.Q_loss), \
            feed_dict={self.x: X, self.learning_rate: learning_rate, self.Q: Q})
        self.iter = self.iter + 1
        return C, reg_ssc, cost_ssc, Q_loss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


class TrainerMLR():
    def __init__(self, model, config):
        self.config = config
        self.model = model

        self.n_input = self.config["n_input"]
        self.batch_size = self.config["batch_size"]

        self.drop_prob = self.config['drop_prob']
        self.diver_param = self.config['diver_param']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']

        self.iter = 0
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.Q = tf.placeholder(tf.float32, [None, None])
        self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.Q_loss, self.Coef, self.D_list = self._build_training_graph()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

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

    def initlization(self):
        # tf.reset_default_graph()
        self.sess.run(self.init)

    def _build_training_graph(self):
        cur_inpu_List, Y_list, x_recon, Coef, D_list = self.model.forward_att(self.x, drop_prob=self.drop_prob,
                                                                              view='1', reuse=False)
        # #重构正则
        recon_ssc = 0
        # recon_ssc=recon_ssc+tf.reduce_sum(tf.pow(tf.subtract(x_recon, self.x), 2.0))
        # #子空间表征正则
        #
        cost_ssc = 0
        count = 0
        for cur, Y, p in zip(cur_inpu_List, Y_list, self.cost_ssc_param):
            # if count==0:
            #     count=count+1
            #     continue
            cost_ssc = cost_ssc + p * tf.reduce_sum(tf.pow(tf.subtract(cur, Y), 2.0))
        #
        # #Q  Multi-Level Representation Learning for Deep Subspace Clustering 有一点点效果但是好像不太高
        Qloss = 0
        # # ord=1
        # # tempQ=tf.matmul(tf.transpose(self.Q), tf.abs(Coef))
        # # Qloss=tf.norm(tempQ, ord)
        #
        # #Self-Supervised Convolutional Subspace Clustering Network
        A = tf.add(tf.transpose(Coef), Coef) / 2
        D = tf.diag(tf.reduce_sum(A, 1))
        Qloss = 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), D - A), self.Q))

        for D in D_list:
            AD = tf.add(tf.transpose(D), D) / 2
            DD = tf.diag(tf.reduce_sum(AD, 1))
            Qloss = Qloss + 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), DD - AD), self.Q))

        diver_cost = 0
        for D in D_list:
            for DJ in D_list:
                if (D != DJ):
                    diver_cost = diver_cost + tf.trace(tf.matmul(D, tf.transpose(DJ)))
                    # self.HSIC(D,DJ)
        #
        #
        # #亲和矩阵正则
        reg_ssc = 0

        reg_ssc = tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D in D_list:
            reg_ssc = reg_ssc + tf.norm(D, 2) ** 2
            #
        loss_ssc = 0
        loss_ssc = cost_ssc * self.cost_ssc_param + self.reg_ssc_param * reg_ssc + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc)
        return optimizer_ssc, loss_ssc, cost_ssc, recon_ssc, reg_ssc, Qloss, Coef, D_list

    def finetune_fit(self, X, learning_rate, Q):
        C, D_list, reg_ssc, cost_ssc, _, Q_loss = self.sess.run(
            (self.Coef, self.D_list, self.reg_ssc, self.cost_ssc, self.optimizer_ssc, self.Q_loss), \
            feed_dict={self.x: X, self.learning_rate: learning_rate, self.Q: Q})
        self.iter = self.iter + 1
        return C, D_list, reg_ssc, cost_ssc, Q_loss

    def infer_clustering(self, X, learning_rate, Q):
        C, D_list, reg_ssc, cost_ssc, Q_loss = self.sess.run(
            (self.Coef, self.D_list, self.reg_ssc, self.cost_ssc, self.Q_loss), \
            feed_dict={self.x: X, self.learning_rate: learning_rate, self.Q: Q})
        self.iter = self.iter + 1
        return C, D_list, reg_ssc, cost_ssc, Q_loss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


class TrainerMLRMV():
    def __init__(self, model, config, loadW=True):

        self.model = model.getModel()

        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.diver_param = self.config['diver_param']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']
        self.IV_param = self.config['IV_param']
        self.lambdaL_param = self.config['lambdaL_param']
        self.loadW = loadW

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

        self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.diver_cost, self.layer_adj_loss, self.IV_cost, \
        self.Q_loss, self.Coef, self.D_list = self._build_training_graph()

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
        recon_ssc = tf.Variable(0, name='recon_ssc', dtype=tf.float32)
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

        diver_cost = 0
        for D_list in D_list_All_view:
            for D in D_list:
                for DJ in D_list:
                    if (D != DJ):
                        diver_cost = diver_cost + tf.trace(tf.matmul(tf.transpose(D), DJ))
                        # diver_cost = diver_cost + tf.reduce_sum(tf.multiply(D, DJ))
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
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 2) ** 2

            #

        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc)
        return optimizer_ssc, loss_ssc, cost_ssc, recon_ssc, reg_ssc_cost, diver_cost, layer_adj_loss, IV_cost, Qloss, Coef_All_view, D_list_All_view

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

        C, D_list, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss = self.sess.run(
            (self.Coef, self.D_list, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
             self.diver_cost, self.layer_adj_loss, self.IV_cost, \
             self.Q_loss), \
            feed_dict=XList_feed)

        self.iter = self.iter + 1

        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss


    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


class TrainerMVDSCN():
    def __init__(self, model, modelC, config, loadW=True):

        self.model = model.getModel()
        self.modelC = modelC.getModel()

        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.model_path = self.config['model_path']

        self.reg_constant_self_loss = self.config['reg_constant_self_loss']
        self.reg_constant_norm_loss_param = self.config['reg_constant_norm_loss']
        self.unify_loss_constant = self.config['unify_loss_constant']
        self.hsic_loss_constant = self.config['hsic_loss_constant']

        self.loadW = loadW
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

        self.optimizer_ssc, self.loss_ssc,self.Coef_Common = self._build_training_graph()

    def initlization(self):
        #
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

    def _build_training_graph(self):

        attH_single_List, attHList_selfexpre_single_List, x_recon_Single_List, Coef_single_List = self._build_training_graph_single()

        selfexpress_loss_single = 0
        for b, a in zip(attH_single_List, attHList_selfexpre_single_List):
            selfexpress_loss_single = selfexpress_loss_single + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        reconst_loss_single = 0
        for b, a in zip(self.xList, x_recon_Single_List):
            reconst_loss_single = reconst_loss_single + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        hsic_loss = 0
        for Coef_single in Coef_single_List:
            for Coef_j_single in Coef_single_List:
                if (Coef_single != Coef_j_single):
                    hsic_loss = hsic_loss + self.HSIC(Coef_j_single, Coef_single)

        reg_loss_single = 0
        for c in Coef_single_List:
            reg_loss_single = reg_loss_single + tf.reduce_sum(tf.pow(c, 2.0))

        # ============================
        attH_Common_List, attH_Common_self_List, x_recon_Common_List, Coef = self._build_training_graph_common()

        selfexpress_loss_common = 0

        for b, a in zip(attH_Common_List, attH_Common_self_List):
            selfexpress_loss_common = selfexpress_loss_common + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        reconst_loss_common = 0
        for b, a in zip(self.xList, x_recon_Common_List):
            reconst_loss_common = reconst_loss_common + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        reg_loss_common = tf.reduce_sum(tf.pow(Coef, 2.0))

        unify_loss = 0
        for Coef_single in Coef_single_List:
            unify_loss = unify_loss + tf.reduce_sum(tf.abs(tf.subtract(Coef, Coef_single)))

        reconst_loss = reconst_loss_single + reconst_loss_common
        reg_loss = reg_loss_single + reg_loss_common
        selfexpress_loss=selfexpress_loss_single+selfexpress_loss_common
        allloss = reconst_loss + selfexpress_loss * self.reg_constant_self_loss + hsic_loss * self.hsic_loss_constant + unify_loss * self.unify_loss_constant + reg_loss*self.reg_constant_norm_loss_param

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(allloss)

        return optimizer, allloss,Coef

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

    def _build_training_graph_single(self):
        attH_single_List = []
        attHList_selfexpre_single_List = []
        x_recon_Single_List = []
        Coef_single_List = []

        for view in range(self.View_num):
            attH = self.model[view].forward_att_encoder(self.xList[view], drop_prob=self.drop_prob, type='single',
                                                        view=str(view + 1), reuse=False)
            attH_single_List.append(attH)

            Coef_single = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32),
                                      name='Coef_' + str(view))
            Coef_single = (Coef_single - tf.diag(tf.diag_part(Coef_single)))

            attH_c_single = tf.matmul(Coef_single, attH)

            attHList_selfexpre_single_List.append(attH_c_single)
            Coef_single_List.append(Coef_single)

            x_recon = self.model[view].forward_att_decoder(attH_c_single, drop_prob=self.drop_prob, type='single',
                                                           view=str(view + 1), reuse=False)
            x_recon_Single_List.append(x_recon)

        return attH_single_List, attHList_selfexpre_single_List, x_recon_Single_List, Coef_single_List

    def _build_training_graph_common(self):
        Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        Coef = (Coef - tf.diag(tf.diag_part(Coef)))

        attH_Common_List = []
        attH_Common_self_List = []
        x_recon_Common_List = []

        for view in range(self.View_num):
            attH_Common = self.modelC[view].forward_att_encoder(self.xList[view], drop_prob=self.drop_prob,type='Common', view=str(view + 1), reuse=False)
            attH_Common_List.append(attH_Common)

            attH_Common_self = tf.matmul(Coef, attH_Common)
            attH_Common_self_List.append(attH_Common_self)

            x_recon_Common = self.modelC[view].forward_att_decoder(attH_Common_self, drop_prob=self.drop_prob,type='Common', view=str(view + 1), reuse=False)
            x_recon_Common_List.append(x_recon_Common)

        return attH_Common_List, attH_Common_self_List, x_recon_Common_List, Coef

        # z1_c = tf.matmul(self.Coef, z1)

    def finetune_fit(self, graph, learning_rate):
        XList = graph.ViewData
        hasW = hasattr(graph, 'W')

        if hasW:
            WList = graph.W

        XList_feed = {}
        for i in range(self.View_num):
            XList_feed["V" + str(i + 1) + ":0"] = XList[i]
            if hasW:
                XList_feed["adj" + str(i + 1) + ":0"] = WList[i]

        XList_feed["learning_rate:0"] = learning_rate

        Coef_Common, _, loss_ssc = self.sess.run((self.Coef_Common,self.optimizer_ssc, self.loss_ssc), feed_dict=XList_feed)

        return loss_ssc,Coef_Common

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")




# 仅仅低维空间中，计算不同视角的共性信息和不同视角单个信息
class TrainerCDMVDSCN():
    def __init__(self, model, config, loadW=True):

        self.model = model.getModel()

        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.model_path = self.config['model_path']

        self.reg_constant_self_loss = self.config['reg_constant_self_loss']
        self.reg_constant_norm_loss_param = self.config['reg_constant_norm_loss']
        self.hsic_loss_constant = self.config['hsic_loss_constant']

        self.loadW = loadW
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

        self.optimizer_ssc, self.loss_ssc,self.Coef_C,self.Coef_single_List = self._build_training_graph()

    def initlization(self):
        #
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

    def _build_training_graph(self):

        attH_single_List, attHList_selfexpre_single_List, x_recon_Single_List, Coef_single_List,Coef_C= self._build_training_graph_single()

        selfexpress_loss_single = 0
        for b, a in zip(attH_single_List, attHList_selfexpre_single_List):
            selfexpress_loss_single = selfexpress_loss_single + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        reconst_loss_single = 0
        for b, a in zip(self.xList, x_recon_Single_List):
            reconst_loss_single = reconst_loss_single + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(b, a), 2.0))

        hsic_loss = 0
        CD_loss=0
        DLoss=0
        for Coef_single in Coef_single_List:

            CD_loss=CD_loss+tf.trace(tf.matmul(tf.transpose(Coef_C),Coef_single))

            for Coef_j_single in Coef_single_List:
                if (Coef_single != Coef_j_single):
                    DLoss = DLoss + tf.trace(tf.matmul(tf.transpose(Coef_single),Coef_j_single))

        hsic_loss=CD_loss+DLoss*0.1

        reg_loss_single = 0
        for c in Coef_single_List:
            reg_loss_single = reg_loss_single + tf.reduce_sum(tf.pow(c, 2.0))

        allloss=reconst_loss_single+self.reg_constant_self_loss*selfexpress_loss_single+hsic_loss*self.hsic_loss_constant+reg_loss_single*self.reg_constant_norm_loss_param

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(allloss)

        return optimizer, allloss,Coef_C,Coef_single_List

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

    def _build_training_graph_single(self):
        attH_single_List = []
        attHList_selfexpre_single_List = []
        x_recon_Single_List = []
        Coef_single_List = []

        Coef_C = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32),
                                  name='Coef_C')

        for view in range(self.View_num):
            attH = self.model[view].forward_att_encoder(self.xList[view], drop_prob=self.drop_prob, type='single',
                                                        view=str(view + 1), reuse=False)
            attH_single_List.append(attH)

            Coef_single = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32),
                                      name='Coef_' + str(view))
            Coef_single = (Coef_single - tf.diag(tf.diag_part(Coef_single)))
            Coef_single_List.append(Coef_single)

            attH_c_single = tf.matmul(Coef_single+Coef_C, attH)

            attHList_selfexpre_single_List.append(attH_c_single)



            x_recon = self.model[view].forward_att_decoder(attH_c_single, drop_prob=self.drop_prob, type='single',
                                                           view=str(view + 1), reuse=False)
            x_recon_Single_List.append(x_recon)

        return attH_single_List, attHList_selfexpre_single_List, x_recon_Single_List, Coef_single_List,Coef_C


    def finetune_fit(self, graph, learning_rate):
        XList = graph.ViewData
        hasW = hasattr(graph, 'W')

        if hasW:
            WList = graph.W

        XList_feed = {}
        for i in range(self.View_num):
            XList_feed["V" + str(i + 1) + ":0"] = XList[i]
            if hasW:
                XList_feed["adj" + str(i + 1) + ":0"] = WList[i]

        XList_feed["learning_rate:0"] = learning_rate

        Coef_C,Coef_single_List, _, loss_ssc = self.sess.run((self.Coef_C,self.Coef_single_List,self.optimizer_ssc, self.loss_ssc), feed_dict=XList_feed)

        return loss_ssc,Coef_C,Coef_single_List

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")



class TrainerNoRawMLRMV():
    def __init__(self, model, config, loadW=True):

        self.model = model.getModel()

        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.diver_param = self.config['diver_param']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']
        self.IV_param = self.config['IV_param']
        self.lambdaL_param = self.config['lambdaL_param']
        self.loadW = loadW

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

        self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.diver_cost, self.layer_adj_loss, self.IV_cost, \
        self.Q_loss, self.Coef, self.D_list = self._build_training_graph()

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
        recon_ssc =0
        for recon,x in zip(x_recon_All_view,self.xList):
            recon_ssc=recon_ssc+tf.reduce_sum(tf.pow(tf.subtract(recon, x), 2.0))
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
                    layer_adj_loss += L * self.get_1st_loss(o, self.adjs[i])

        IV_cost = 0
        for Coef in Coef_All_view:
            for Coef_j in Coef_All_view:
                if (Coef != Coef_j):
                    IV_cost = IV_cost + self.HSIC(Coef_j, Coef)
                    # IV_cost = IV_cost + tf.reduce_sum(tf.multiply(Coef_j, Coef))

        diver_cost = 0
        for D_list in D_list_All_view:
            for D in D_list:
                for DJ in D_list:
                    if (D != DJ):
                        diver_cost = diver_cost + tf.trace(tf.matmul(tf.transpose(D), DJ))
                        # diver_cost = diver_cost + tf.reduce_sum(tf.multiply(D, DJ))
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
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 2) ** 2

            #

        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc)
        return optimizer_ssc, loss_ssc, cost_ssc, recon_ssc, reg_ssc_cost, diver_cost, layer_adj_loss, IV_cost, Qloss, Coef_All_view, D_list_All_view

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

        C, D_list, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss = self.sess.run(
            (self.Coef, self.D_list, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
             self.diver_cost, self.layer_adj_loss, self.IV_cost, \
             self.Q_loss), \
            feed_dict=XList_feed)

        self.iter = self.iter + 1

        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss


    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")



class TrainerCLDMLRMV():
    def __init__(self, model, config, loadW=True):
        self.zeroconstant = tf.constant(0, dtype=tf.float32)

        self.model = model.getModel()

        self.config = config

        self.batch_size = self.config["batch_size"]
        self.View_num = self.config["View_num"]
        self.dims = self.config["dims"]
        self.drop_prob = self.config['drop_prob']
        self.diver_param = self.config['diver_param']
        self.cost_ssc_param = self.config['cost_ssc_param']
        self.reg_ssc_param = self.config['reg_ssc_param']
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']
        self.IV_param = self.config['IV_param']
        self.lambdaL_param = self.config['lambdaL_param']
        self.loadW = loadW

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

        self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.diver_cost, self.layer_adj_loss, self.IV_cost, \
        self.Q_loss, self.Coef, self.D_list,self.CoefALL = self._build_training_graph()

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
        with tf.variable_scope('decoder_encoder_coef', reuse=tf.AUTO_REUSE):
            CoefALL = tf.get_variable('Coef_ALL_View', [self.batch_size, self.batch_size])  # 此处使用外围的initializer=tf.constant_initializer(0.8)初始化（类似继承）

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

        layer_adj_loss = self.zeroconstant

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

        diver_cost = 0
        for D_list in D_list_All_view:
            for D in D_list:
                for DJ in D_list:
                    if (D != DJ):
                        diver_cost = diver_cost + tf.trace(tf.matmul(tf.transpose(D), DJ))
                        # diver_cost = diver_cost + tf.reduce_sum(tf.multiply(D, DJ))
                    # self.HSIC(D,DJ)

        Qloss = 0
        # # ord=1
        # # tempQ=tf.matmul(tf.transpose(self.Q), tf.abs(Coef))
        # # Qloss=tf.norm(tempQ, ord)
        #
        # #Self-Supervised Convolutional Subspace Clustering Network

        # for Coef in Coef_All_view:
        #     A = tf.add(tf.transpose(Coef), Coef) / 2
        #     D = tf.diag(tf.reduce_sum(A, 1))
        #
        #     Qloss = Qloss + 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), D - A), self.Q))

        # for D_list in D_list_All_view:
        #     for D in D_list:
        #         AD = tf.add(tf.transpose(D), D) / 2
        #         DD = tf.diag(tf.reduce_sum(AD, 1))
        #         Qloss = Qloss + 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Q), DD - AD), self.Q))

        #
        #
        # #亲和矩阵正则
        reg_ssc_cost = 0
        for Coef in Coef_All_view:
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 2) ** 2


        reg_ssc_cost=reg_ssc_cost+tf.norm(CoefALL,1)

            #

        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

        optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_ssc)
        return optimizer_ssc, loss_ssc, cost_ssc, recon_ssc, reg_ssc_cost, diver_cost, layer_adj_loss, IV_cost, Qloss, Coef_All_view, D_list_All_view,CoefALL

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

        C, D_list,CoefALL, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss = self.sess.run(
            (self.Coef, self.D_list,self.CoefALL, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
             self.diver_cost, self.layer_adj_loss, self.IV_cost, \
             self.Q_loss), \
            feed_dict=XList_feed)

        self.iter = self.iter + 1

        return C, D_list,CoefALL, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")



class TrainerMLRMVGaussDistribution():
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
        self.model_path = self.config['model_path']
        self.Q_param = self.config['Q_param']
        self.IV_param = self.config['IV_param']
        self.lambdaL_param = self.config['lambdaL_param']
        self.loadW = loadW

        self.dis=[]

        for i in range(self.View_num):
                self.dis.append(Discriminator(V=i+1))

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
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 2) ** 2


            #
        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

        d_fake_list=[]
        d_real_list=[]
        for cur,disc in zip(cur_inpu_List_All_view,self.dis):
            curInput = cur[-1]

            d_fake=disc.construct(curInput, reuse=False)

            prior = tf.random_normal(shape=tf.shape(curInput))

            d_real=disc.construct(prior, reuse=True)

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

        C, D_list, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,disLoss = self.sess.run(
            (self.Coef, self.D_list, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
             self.diver_cost, self.layer_adj_loss, self.IV_cost, \
             self.Q_loss,self.disLoss), \
            feed_dict=XList_feed)

        for j in range(2):
            _,genLoss=self.sess.run([self.optimizer_ssc_disc_var,self.genLoss], feed_dict=XList_feed)

        self.iter = self.iter + 1

        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


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
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 1)


            #
        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss

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

        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


class TrainerMLRMVGaussUniversalDistribution_AutoGraph():
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
        self.VW = tf.placeholder(tf.float32, [self.View_num], name="VW")

        self.optimizer,self.optimizer_ssc,self.optimizer_ssc_disc_var, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc, self.diver_cost, self.layer_adj_loss, self.IV_cost, \
        self.Q_loss, self.Coef, self.D_list,self.genLoss,self.disLoss,self.FCon,self.Orloss,self.consensus_loss = self._build_training_graph()




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
        FCon = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32),
                             name='Coef_Consensus')  # 创建公共表征
        FCon = FCon - tf.diag(tf.diag_part(FCon))

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
            reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 1)  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))
            # reg_ssc_cost = reg_ssc_cost + tf.norm(Coef, 2) ** 2  # tf.sqrt( tf.reduce_sum(tf.pow(Coef, 2)))

        for D_list in D_list_All_view:
            for D in D_list:
                reg_ssc_cost = reg_ssc_cost + tf.norm(D, 1)


        #consensus loss
        consensus_loss=0
        #Coef_All_view
        for D_list,Coef,index_vw in zip(D_list_All_view,Coef_All_view,range(self.View_num)):
            CD=Coef
            for D in D_list:
                CD=Coef+D

            consensus_loss = consensus_loss + self.VW[index_vw]*tf.reduce_sum(tf.subtract(FCon, CD) ** 2)  # tf.matmul(FCon, Coef)

        #D_list_All_view

        Orloss=0
        Orloss=tf.subtract(tf.matmul(FCon,tf.transpose(FCon)),tf.eye(self.batch_size))



            #
        loss_ssc = cost_ssc + self.reg_ssc_param * reg_ssc_cost + recon_ssc + Qloss * self.Q_param + self.diver_param * diver_cost + IV_cost * self.IV_param + layer_adj_loss+consensus_loss

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

        return optimizer,optimizer_ssc,optimizer_ssc_disc_var, loss_ssc, cost_ssc, recon_ssc, reg_ssc_cost, diver_cost, layer_adj_loss, IV_cost, Qloss, Coef_All_view, D_list_All_view,genLoss,disLoss,FCon,Orloss,consensus_loss

    def finetune_fit(self, graph, learning_rate, Q,VW):
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
        XList_feed["VW:0"] = VW



        for j in range(1):
            _= self.sess.run([self.optimizer], feed_dict=XList_feed)

        for j in range(1):
            C, D_list, _, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,disLoss,FCon = self.sess.run(
                (self.Coef, self.D_list, self.optimizer_ssc, self.loss_ssc, self.cost_ssc, self.recon_ssc, self.reg_ssc,
                 self.diver_cost, self.layer_adj_loss, self.IV_cost, \
                 self.Q_loss,self.disLoss,self.FCon), \
                feed_dict=XList_feed)

        for j in range(2):
            _,genLoss=self.sess.run([self.optimizer_ssc_disc_var,self.genLoss], feed_dict=XList_feed)

        self.iter = self.iter + 1

        return C, D_list, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss,FCon

    def save_model(self):
        mkdir(self.model_path)
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")

