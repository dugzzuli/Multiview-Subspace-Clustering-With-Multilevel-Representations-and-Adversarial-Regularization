import argparse
import yaml
import numpy as np
import linecache
import  numpy as np
import pandas as pd
from pylab import *
from matplotlib.font_manager import FontProperties

def plotPic(names,y,datase_name_list,type="ACC",dirreuslt="../results/pic/",linestyle=1,floorD=0.4,flag='β'):
    """

    :param names:
    :param y:
    :param datase_name_list:
    :param type:
    :param dirreuslt:
    :param linestyle:
    :param floorD:
    :return:
    """

    x = range(len(names))
    plt.figure(figsize=(13, 9), facecolor='#FFFFFF')  # 绘制具有一行两列的折线图figsize设置画布的大小  figsize=[8,6],

    plt.rc('font', family='Times New Roman')

    legend_list=[]

    index=0
    markerList=['o','d','^','>','<','h']
    lineTypeList=['-','--',':',':.']
    colorList=['b','r','g','c','m','y','k','w']
    for y_index,data_name in zip(y,datase_name_list):
        p1, = plt.plot(x, y_index, lineTypeList[linestyle], linewidth=5, marker=markerList[index], color=colorList[index],ms=17, mfc=colorList[index], label=u'{}'.format(data_name))
        legend_list.append(p1)
        index=index+1

    # legend_list=[p1,p2,p3,p4,p5]

    font = {'family': 'Times new roman',
            'weight': 'normal',
            'size': 34}
    font1 = {'family': 'Times new roman',
             'weight': 'normal',
             'size': 45}
    # plt.legend(prop=font, loc = 3)  # 让图例生效 #

    length = len(legend_list)
    # first_legend = plt.legend(handles=legend_list[0:int(length/2)+1], prop=font, loc=3)
    # # Add the legend manually to the current Axes.
    # ax = plt.gca().add_artist(first_legend)
    # # Create another legend for the second line.
    # plt.legend(handles=legend_list[int(length/2)+1:], prop=font, loc=4)
    plt.legend(handles=legend_list, prop=font, loc=4)

    plt.tick_params(labelsize=14)  # 9
    plt.tick_params(length=8)  # 9
    plt.tick_params(width=4)

    # plt.xlim(0, 6)
    plt.ylim(floorD, 1.0)
    plt.xticks(x, names, rotation=0, size=40)  # rotation表示旋转的角度 40 or 55

    plt.yticks(np.arange(floorD,1.1,0.1), size=40)  # np.arange(0.5, 1.0, 0.05)
    plt.margins(0)  # 去除白边s
    plt.subplots_adjust(left=0.12, bottom=0.12)

    # 设置中文字体
    # font2 = FontProperties(fname=r"c:\windows\fonts\Times New Roman.ttf", size=55)  #size = 12
    # plt.xlabel(u"α", fontdict=font1)  # X轴标签γ
    plt.xlabel(u"{}".format(flag), fontdict=font1)  # X轴标签βγ
    # plt.xlabel(u"γ", fontdict=font1)  # X轴标签βγ
    # plt.ylabel(u"Micro-F1", fontdict=font1)  # Y轴标签
    plt.ylabel(u"{}".format(type), fontdict=font1)  # Y轴标签
    # plt.ylabel(u"Accuracy of clustering", fontproperties=font2)  # Y轴标签
    # plt.title("(a) ")  #z 标题

    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(4)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(4)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(4)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(4)  ####设置上部坐标轴的粗细
    plt.tight_layout()
    plt.savefig('{}{}.png'.format(dirreuslt,type))
    plt.show()


if __name__ == '__main__':
    type = "ACC"
    datase_name_list=['MSRCv1','ORL_mtv']
    nmiList=[]
    accList=[]
    for datase_name in datase_name_list:
        parser = argparse.ArgumentParser(description='Deep subspace')
        parser.add_argument('--dataset', default=datase_name, type=str, help='dataset')
        args = parser.parse_args()
        print(args)
        dataset_name = args.dataset.replace('\n', '').replace('\r', '')
        learning_rate='0.0005'
        d_a = yaml.load(open("../configGaussparameter.yaml", 'r'))

        dirP = '../baseline/configGaussparameter/{}/{}/{}/'.format('regularization', dataset_name, learning_rate)

        file=dirP+"/data.txt"
        alllines=linecache.getlines(file)

        resultsacc=[]
        resultsnmi = []

        for line in alllines:
            if(line.find("bestAcc")!=-1):
                arrs=line.split('\t')
                acc=float(arrs[0].split(":")[1].strip()[:6])
                nmi=float(arrs[1].split(":")[1].strip()[:6])

                resultsacc.append(acc)
                resultsnmi.append(nmi)

        accList.append(resultsacc)
        nmiList.append(resultsnmi)
    if(type=="ACC"):
        y = accList
    else:
        y = nmiList

    names = ['0.001','0.01','0.1','1','10','100']  # 固定Beta, Beta
    x = range(len(names))

    plt.figure(figsize=(13, 9), facecolor='#FFFFFF')  # 绘制具有一行两列的折线图figsize设置画布的大小  figsize=[8,6],

    y1 = y[0]
    y2 = y[1]

    # y6 = [0.3388, 0.4326, 0.4368, 0.4269, 0.3103, 0.4499, 0.3044]  # Rochester

    p1, = plt.plot(x, y[0], '--', linewidth=5, marker='o', ms=17, mfc='w', label=u'{}'.format(datase_name_list[0]))
    p2, = plt.plot(x, y[1], '--', linewidth=5, marker='d', ms=17, label=u'{}'.format(datase_name_list[1]))

    font = {'family': 'Times new roman',
            'weight': 'normal',
            'size': 34}
    font1 = {'family': 'Times new roman',
             'weight': 'normal',
             'size': 45}
    # plt.legend(prop=font, loc = 3)  # 让图例生效 #

    # first_legend = plt.legend(handles=[p1, p2, p3], prop=font, loc=3)
    # # Add the legend manually to the current Axes.
    # ax = plt.gca().add_artist(first_legend)
    # # Create another legend for the second line.
    plt.legend(handles=[p1, p2], prop=font, loc=4)

    plt.tick_params(labelsize=14)  # 9
    plt.tick_params(length=8)  # 9
    plt.tick_params(width=4)
    # plt.xlim(0, 6)
    plt.ylim(0.40, 1.0)
    plt.xticks(x, names, rotation=0, size=40)  # rotation表示旋转的角度 40 or 55

    plt.yticks([0.40, 0.50, 0.60, 0.7, 0.8, 0.9, 1.0], size=40)  # np.arange(0.5, 1.0, 0.05)
    plt.margins(0)  # 去除白边s
    plt.subplots_adjust(left=0.16, bottom=0.16)

    # 设置中文字体
    # font2 = FontProperties(fname=r"c:\windows\fonts\Times New Roman.ttf", size=55)  #size = 12
    # plt.xlabel(u"α", fontdict=font1)  # X轴标签γ
    # plt.xlabel(u"β", fontdict=font1)  # X轴标签βγ
    plt.xlabel(u"γ", fontdict=font1)  # X轴标签βγ
    # plt.ylabel(u"Micro-F1", fontdict=font1)  # Y轴标签
    plt.ylabel(u"{}".format(type), fontdict=font1)  # Y轴标签
    # plt.ylabel(u"Accuracy of clustering", fontproperties=font2)  # Y轴标签
    # plt.title("(a) ")  #z 标题

    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(4)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(4)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(4)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(4)  ####设置上部坐标轴的粗细

    plt.savefig('{}_2_{}.png'.format(dirP, type))
    plt.show()



