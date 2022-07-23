import argparse
import yaml
import numpy as np
import linecache


def drawParameter(type,data):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 构造需要显示的值
    X = np.arange(0, 6, step=1)  # X轴的坐标
    Y = np.arange(0, 6, step=1)  # Y轴的坐标
    # 设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
    Z = data
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.zeros_like(X)  # 设置柱状图的底端位值
    Z = Z.ravel()  # 扁平化矩阵
    width = height = 0.55  # 每一个柱子的长和宽
    cnames = {
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'beige': '#F5F5DC',
        'bisque': '#FFE4C4',
        'black': '#000000',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'floralwhite': '#FFFAF0',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'ghostwhite': '#F8F8FF',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'honeydew': '#F0FFF0',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'ivory': '#FFFFF0',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightcyan': '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen': '#90EE90',
        'lightgray': '#D3D3D3',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lightyellow': '#FFFFE0',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mintcream': '#F5FFFA',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navajowhite': '#FFDEAD',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#FAA460',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32'}
    colors = plt.cm.jet(Z.flatten() / float(Z.max()))
    colors = [cnames['yellowgreen'], cnames['yellow'], cnames['white'], cnames['wheat'], cnames['violet'],
              cnames['teal'], cnames['teal']]
    colorsArr = []
    flag = 0
    for i in range(36):

        if (i <= 5 and i >= 0):
            colorsArr.append(colors[0])

        if (i <= 11 and i >= 6):
            colorsArr.append(colors[1])

        if (i <= 17 and i >= 12):
            colorsArr.append(colors[2])

        if (i <= 23 and i >= 18):
            colorsArr.append(colors[3])

        if (i <= 29 and i >= 24):
            colorsArr.append(colors[4])

        if (i <= 35 and i >= 30):
            colorsArr.append(colors[5])
    # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    # ax.spines['bottom'].set_linewidth(4)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(4)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(4)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(4)  ####设置上部坐标轴的粗细
    font = {'family': 'Times new roman',
            'weight': 'normal',
            'size': 10}
    font1 = {'family': 'Times new roman',
             'weight': 'normal',
             'size': 15}

    ax.set_zticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    ax.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax.set_zticklabels(['0.0','0.1', '0.2','0.3', '0.4','0.5', '0.6','0.7', '0.8','0.9' ])

    ax.bar3d(X, Y, bottom, width, height, Z, color=colorsArr, shade=True)  #
    # 坐标轴设置
    ax.set_xlabel(u"α", fontdict=font1)
    ax.set_ylabel(u"β", fontdict=font1)
    ax.set_zlabel(type)

    plt.tight_layout()
    # plt.margins(0)  # 去除白边s
    # plt.subplots_adjust(left=0.16, bottom=0.16)

    plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(dirP + '../{}_{}.png'.format(dataset_name,type))
    plt.show()


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Deep subspace')
    parser.add_argument('--dataset', default='MSRCv1', type=str, help='dataset')
    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset.replace('\n', '').replace('\r', '')
    learning_rate='0.0005'
    d_a = yaml.load(open("../configGaussparameter.yaml", 'r'))

    dirP = '../baseline/configGaussparameter/{}/{}/{}/'.format('result_param', dataset_name, learning_rate)

    file=dirP+"/data.txt"
    alllines=linecache.getlines(file)

    resultsacc=[]
    resultsnmi = []

    for line in alllines:
        if(line.find("bestAcc")!=-1):
            arrs=line.split(',')
            acc=float(arrs[0].split(":")[1].strip()[:6])
            nmi=float(arrs[1].split(":")[1].strip()[:6])

            resultsacc.append(acc)
            resultsnmi.append(nmi)

    accm=np.array(resultsacc).reshape(6,6)
    nmim = np.array(resultsnmi).reshape(6, 6)

    drawParameter("ACC",accm)
    drawParameter("NMI",nmim)




