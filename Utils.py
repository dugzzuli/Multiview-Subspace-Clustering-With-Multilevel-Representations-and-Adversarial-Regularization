
import numpy as np

from scipy.sparse.linalg import svds

from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import tensorflow as tf

from bestMap import bestmap
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

from embedding_train.utilsDugking import multi_label_classification


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def mkdir(path):

    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:

        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    # _,n=np.shape(C)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)

    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score
def post_clustering(C,alpha,num_classes,label,d, ro,flag=True,graph=None):


    C = thrC(C, alpha)
    y_x, CKSym_x = post_proC(C, num_classes, d, ro)

    if(graph!=None):
        with(open('embedding_train/embeddingResults/{}_MvSC-MRAR.txt'.format(graph.datasetname), 'a+')) as f:
            test_ratio =[0.6,0.7,0.8,0.9] #np.arange(0.5, 1.0, 0.2)
            dane = []
            for tr in test_ratio[-1::-1]:
                print('============train ration-{}=========='.format(1 - tr))
                micro, macro = multi_label_classification(C, graph.Y, tr)
                dane.append('{:.4f}'.format(micro) + ' & ' + '{:.4f}'.format(macro))
            print(' & '.join(dane))
            f.write((' & '.join(dane)))
            f.write('\n')
            f.flush()

    # np.save('save_embedding',CKSym_x)
    # if(flag):
    #     # Display matrix
    #     plt.matshow(CKSym_x)
    #
    #     plt.show()
    missrate_x = err_rate(label, y_x)
    acc = 1 - missrate_x
    nmire = nmi(label, y_x)
    mapY=bestmap(label,y_x)
    return acc,nmire,mapY

def getNMIAndACC(label, y_x):
    missrate_x = err_rate(label, y_x)
    acc = 1 - missrate_x
    nmire = nmi(label, y_x)
    mapY = bestmap(label, y_x)
    return acc, nmire, mapY



import numpy as np
def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)

def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

if (__name__ == "__main__"):
    data=np.loadtxt('results.csv',delimiter=',',dtype=int)
    for i in range(100):
        row=data[i,:-3]+1
        print("Row({}):{}".format(i,calc_ent(row)))



