import numpy as np
import scipy.io as sio
import matplotlib.pylab as plt
if __name__ == '__main__':
    dataset = 'DiMSC'
    data = sio.loadmat('./Visualization/{}.mat'.format(dataset))

    # arr=data['CKSym'][0]
    # sum=np.zeros_like(arr[0])
    # for i in range(5):
    #     sum=sum+arr[i]
    #
    # plt.matshow(sum)

    plt.matshow(data['CKSym'])

    plt.savefig('./PIC/{}.png'.format(dataset))

    plt.show()



