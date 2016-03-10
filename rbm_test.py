from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix
import csv

def scaleRows(mat_in):

    mat=np.zeros(mat_in.shape)

    for i in xrange(mat_in.shape[0]):
        if max(mat_in[i])>0:
            mat[i]=np.divide(mat_in[i],max(mat_in[i]))

    return mat

def sigmoid(x):
    out=np.divide(1.0,(1.0+np.exp(-x)))
    return out

def readMat(filename):
    mat=[]
    start=True
    rownames=[]
    colnames=[]
    for line in csv.reader(open(filename,'r'),delimiter='\t'):
        if start:
            colnames=line[1:]
            start=False
        else:
            mat.append([float(x) for x in line[1:]])
            rownames.append(line[0])

    mat_out=np.asarray(mat)
    return (mat_out,rownames, colnames)

(dat, rownames, colnames)=readMat('/Users/danielcarlin/Data/GTEx/test_train/train_1.tab')

dat_scaled=scaleRows(dat)

class RestrictedBoltzmannMachine:
    """container for RBM methods"""
    def __init__(self, dat):
        self.trainRBM(dat)

    def trainRBM(self, dat):
        # training parameters
        epsilon = 0.1
        momentum = 0.9

        num_epochs = 30
        batch_size = 128
        num_batches = dat.shape[1]//batch_size

        # model parameters
        num_vis = dat.shape[0]
        num_hid = 60

        # initialize weights
        #w_vh = cm.CUDAMatrix(0.1 * np.random.randn(num_vis, num_hid))
        #w_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
        #w_h = cm.CUDAMatrix(-4.*np.ones((num_hid, 1)))
        w_vh = 0.1 * np.random.randn(num_vis, num_hid)
        w_v = np.zeros((num_vis, 1))
        w_h = np.ones((num_hid, 1))

        # initialize weight updates
        wu_vh = np.zeros((num_vis, num_hid))
        wu_v = np.zeros((num_vis, 1))
        wu_h = np.zeros((num_hid, 1))

        # initialize temporary storage
        v = np.zeros((num_vis, batch_size))
        h = np.zeros((num_hid, batch_size))
        self.hidden=np.zeros((num_hid,dat.shape[1]))

        for epoch in range(num_epochs):
            print("Epoch %i" % (epoch + 1))
            err = []

            for batch in range(num_batches):
                print("batch %i" % (batch+1))
                # get current minibatch
                v = dat[:,range(batch*batch_size,(batch + 1)*batch_size)]

                # apply momentum
                wu_vh=wu_vh*momentum
                wu_v=wu_v*momentum
                wu_h=wu_h*momentum

                # positive phase
                h=sigmoid(np.dot(w_vh.T, v)+(w_h*np.ones((1,batch_size))))

                wu_vh=wu_vh+np.dot(v,h.T)
                wu_v=wu_v+np.sum(v, axis = 1).reshape(num_vis,1)
                wu_h=wu_h+np.sum(h, axis = 1).reshape(num_hid,1)

                # sample hiddens

                h=(h<np.random.uniform(size=(num_hid, batch_size)).astype(int))

                # negative phase
                v=sigmoid(np.dot(w_vh,h)+(w_v*np.ones((1,batch_size))))

                #here h is h_tilde
                h=sigmoid(np.dot(w_vh.T, v)+(w_h*np.ones((1,batch_size))))

                wu_vh=wu_vh-np.dot(v, h.T)
                wu_v=wu_v-np.sum(v, axis = 1).reshape(num_vis,1)
                wu_h=wu_h-np.sum(h, axis = 1).reshape(num_hid,1)

                # update weights
                w_vh=w_vh+(wu_vh*(epsilon/batch_size))
                w_v=w_v+(wu_v*(epsilon/batch_size))
                w_h=w_h+(wu_h*(epsilon/batch_size))

                # calculate reconstruction error
                this_err=sum((v-dat[:,range(batch*batch_size,(batch + 1)*batch_size,1)])**2)/(num_vis*batch_size)
                err.append(this_err)
                self.hidden[:,range(batch*batch_size,(batch + 1)*batch_size,1)]=h

            print("Mean squared error: %f" % np.mean(err))
        self.w_vh=w_vh
        self.w_v=w_v
        self.w_h=w_h

    def testError(self,test_dat):



rbm = RestrictedBoltzmannMachine(dat_scaled)