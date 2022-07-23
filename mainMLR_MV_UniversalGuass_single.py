#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from Dataset.dataset import Dataset
from Utils import mkdir, post_clustering, getNMIAndACC
import random
import warnings
from TrainerD.Trainer import TrainerMLRMVGaussUniversalDistribution
from pretrainer import PreTrainer
from Model import MVGaussModel
from Visualizer import Visualizer
import yaml
import settings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

nmi = normalized_mutual_info_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i,j-1] = 1 
        
    return Q

def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0],1])
        Theta[i, :] = 1/2*np.sum(np.square(Q - Qq), 1)

    return Theta

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep subspace')
    parser.add_argument('--dataset', default='BBCSport', type=str, help='dataset')
    args = parser.parse_args()
    print(args)
    
    initialize=True

    d_a = yaml.safe_load(open("configUniversalGauss_tnnls.yaml", 'r'))

    dataset_name = args.dataset.replace('\n', '').replace('\r', '')

    layers = d_a[dataset_name]['layers']
    View_num = d_a[dataset_name]['View_num']
    ft_times =1000 # d_a[dataset_name]['ft_times']

    random.seed(9001)
    dataset_config = {
        'View': ['./Database/' + dataset_name + '/View{}.txt'.format(i + 1) for i in range(View_num)],
        'W': ['./Database/' + dataset_name + '/W{}.txt'.format(i + 1) for i in range(View_num)],
        'label_file': './Database/' + dataset_name + '/group.txt'}

    loadW = d_a[dataset_name]['loadW']
    graph = Dataset(dataset_config, loadW)
    graph.datasetname=dataset_name
    if(dataset_name== 'Caltech101-7') or (dataset_name== 'Reuters'):
        pass
        print("no norm")
    else:
        graph.normData()
        pass

    label = np.argmax(graph.Y, 1)
    print(graph)
    alpha = max(0.4 - (graph.num_classes - 1) / 10 * 0.1, 0.1)

    dims = [np.shape(vData)[1] for vData in graph.ViewData]

    batch_size = graph.num_nodes

    pretrain_config = {
        'View': layers,
        'batch_size': batch_size,
        'pretrain_params_path': './Log/' + dataset_name + '/pretrain_params.pkl'}

    if d_a[dataset_name]['pretrain']:
        pretrainer = PreTrainer(pretrain_config)
        for i in range(len(graph.ViewData)):
            print("pretrain {}".format('V' + str(i + 1)))
            pretrainer.pretrain(graph.ViewData[i], 'V' + str(i + 1))

    for learning_rate in d_a[dataset_name]['learning_rate']:
        dirP = './baseline/UniversalGuass/{}/{}/{}/'.format(d_a['resultDirName'],dataset_name, learning_rate)
        mkdir(dirP)
        with open(dirP + '{}_data.txt'.format(initialize), 'a+') as f:
            
            for cost_ssc_param in d_a[dataset_name]['cost_ssc_param']:
                for reg_ssc_param in d_a[dataset_name]['reg_ssc_param']:
                    for diver_param in d_a[dataset_name]['diver_param']:
                        for Q_param in d_a[dataset_name]['Q_param']:
                            for reg_ssc_param_2 in d_a[dataset_name]['reg_ssc_param_2']:
                                for lambdaL_param in d_a[dataset_name]['lambdaL_param']:
                                    f.write("*********************"*5)
                                    f.write("\n")
                                    for count in range(5):
                                        IV_param=diver_param
                                        
                                        tf.reset_default_graph()
                                        
                                        model_config = {'att_shape': layers,
                                                        'batch_size': batch_size,
                                                        'is_init': initialize,
                                                        'View_num': View_num,
                                                        'pretrain_params_path': './Log/{}/pretrain_params.pkl'.format(
                                                            dataset_name),
                                                        }

                                        model = MVGaussModel(model_config)

                                        Q = np.zeros((batch_size, graph.num_classes))
                                        env = "version:{}_dataset_name:{}_lambdaL_param:{}_diver_param:{}_cost_ssc_param:{}_reg_ssc_param:{}_reg_ssc_param_2:{}_Q_param:{}_IV_param:{}_layer:{}".format(d_a[dataset_name]['version'], dataset_name,
                                                                            '_'.join(list(map(str, lambdaL_param))),
                                                                            diver_param,
                                                                            '_'.join(list(map(str, cost_ssc_param))),
                                                                            reg_ssc_param,reg_ssc_param_2, Q_param,
                                                                            IV_param,layers)
                                        # print()
                                        trainer_config = {
                                            'batch_size': batch_size,
                                            'drop_prob': 0.2,
                                            'dims': dims,
                                            'lambdaL_param': lambdaL_param,
                                            'View_num': View_num,
                                            'diver_param': diver_param,
                                            'cost_ssc_param': cost_ssc_param,
                                            'reg_ssc_param_2': reg_ssc_param_2,
                                            'reg_ssc_param': reg_ssc_param,
                                            'Q_param': Q_param,
                                            'IV_param': IV_param,
                                            'model_path': './Log/{}/{}_model.pkl'.format(dataset_name, dataset_name),
                                        }
                                        vis = None
                                        if (d_a['showVisualdom']):
                                            vis = Visualizer(env)

                                        trainer = TrainerMLRMVGaussUniversalDistribution(model, trainer_config, loadW)

                                        trainer.initlization()
                                        if (d_a[dataset_name]['isRestore']):
                                            trainer.restore_model()

                                        accList=[]
                                        nmiList=[]
                                        for iter_ft in range(ft_times):
                                            showResults = {}
                                            showCosts = {}
                                            results_label = None
                                            try:

                                                C_ALL_view, D_list_ALL_view, loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss = trainer.finetune_fit(
                                                    graph, learning_rate, Q)

                                                # C = thrC(C+1./3*np.sum(D_list), alpha)
                                                CALL = np.zeros_like(C_ALL_view[0])

                                                for C, D_list, view in zip(C_ALL_view, D_list_ALL_view, range(View_num)):
                                                    l = len(D_list)
                                                    DALL = np.zeros_like(D_list[0])
                                                    for D1 in D_list:
                                                        DALL = DALL + D1

                                                    IsD = d_a[dataset_name]['IsD']
                                                    IsDDiverse = d_a[dataset_name]['IsDDiverse']
                                                    CALL = CALL + C * (IsD) + DALL * (IsDDiverse) / l

                                                    # print(np.sum(CALL))

                                                    try:
                                                        if d_a['showEveryViewResult'] and (iter_ft + 1) % 30 == 0 or (iter_ft == ft_times - 1):  #
                                                            tempc = C * (IsD) + DALL * (IsDDiverse) / l
                                                            acc_single, nmire_single, y_x_single = post_clustering(tempc,
                                                                                                                   alpha,
                                                                                                                   graph.num_classes,
                                                                                                                   label,
                                                                                                                   d_a[
                                                                                                                       dataset_name][
                                                                                                                       'd'],
                                                                                                                   d_a[
                                                                                                                       dataset_name][
                                                                                                                       'ro'],graph=graph)
                                                            y_x_single = np.reshape(y_x_single, [-1, 1])
                                                            if (view == 0):
                                                                results_label = y_x_single
                                                            else:
                                                                results_label = np.concatenate([results_label, y_x_single],
                                                                                               axis=1)

                                                            result_single = 'View:{},acc:{},nmi:{}'.format(view, acc_single,
                                                                                                           nmire_single)

                                                            print(result_single)

                                                    except Exception as e:
                                                        print(e)
                                                        pass
                                                    finally:
                                                        pass

                                                acc, nmire, y_x = post_clustering(CALL, alpha, graph.num_classes, label,
                                                                                  d_a[dataset_name]['d'],
                                                                                  d_a[dataset_name]['ro'],graph=graph)

                                                y_x = np.reshape(y_x, [-1, 1])
                                                if (results_label is not None):
                                                    results_label = np.concatenate([results_label, y_x], axis=1)
                                                else:
                                                    results_label = y_x


                                            except Exception as e:
                                                print(e)

                                            finally:
                                                if (iter_ft + 1) % 1== 0:
                                                    print(env)
                                                    print("\n")
                                                    print("迭代次数....{}".format(iter_ft))

                                                    results = "loss_ssc(总损失):{}, cost_ssc(子空间损失):{}, recon_ssc(重构损失):{}, reg_ssc(正则损失):{},\n diver_cost(多样性损失):{},layer_adj_loss(层图正则损失):{},IV_cost(子空间之间):{},Q_loss(伪标签损失):{},\n acc:{},nmi:{}".format(
                                                        loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss,
                                                        IV_cost, Q_loss, acc, nmire)
                                                    
                                                    accList.append(acc)
                                                    nmiList.append(nmire)
                                                    
                                                    tempa=max(accList)
                                                    tempn = max(nmiList)
                                                    maxstr="ecpoch:{},bestAcc:{}\tbestNMI:{}\n".format(accList.index(tempa),tempa,tempn)
                                                    print(maxstr)
                                                    
                                                    

                                                    f.flush()



                                                    print(results)
                                                    if (vis is not None):
                                                        showResults["acc"] = acc
                                                        showResults["nmire"] = nmire

                                                        loss_ssc, cost_ssc, recon_ssc, reg_ssc, diver_cost, layer_adj_loss, IV_cost, Q_loss,genLoss,disLoss

                                                        showCosts["genLoss"] = loss_ssc
                                                        showCosts["disLoss"] = cost_ssc

                                                        # showCosts["loss_ssc"] = loss_ssc
                                                        # showCosts["cost_ssc"] = cost_ssc
                                                        # showCosts["recon_ssc"] = recon_ssc
                                                        # showCosts["reg_ssc"] = reg_ssc
                                                        # showCosts["diver_cost"] = diver_cost
                                                        # showCosts["layer_adj_loss"] = layer_adj_loss
                                                        # showCosts["IV_cost"] = IV_cost
                                                        # showCosts["Q_loss"] = Q_loss

                                                        vis.plot_many_stack(showResults)
                                                        vis.plot_many_stack(showCosts)

                                                if (iter_ft + 1) % d_a[dataset_name]['T'] == 0:
                                                    try:

                                                        if y_x is not None:
                                                            Q = np.eye(graph.num_classes)[
                                                                np.array(y_x.astype(int)).reshape(-1)]
                                                    except Exception as e:
                                                        pass
                                                    finally:
                                                        pass

                                               
                                                    
                                        if(d_a['isSaveModel']):
                                            trainer.save_model()

                                        if d_a[dataset_name]['isS']:
                                            for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                                                for ro in [1, 2, 3, 4, 5, 6, 7, 8]:
                                                    acc, nmire, y_x = post_clustering(CALL, alpha, graph.num_classes, label,
                                                                                      d, ro)
                                                    print("d:{} ro:{} acc:{} nmire:{}".format(d, ro, acc, nmire))