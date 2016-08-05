#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy
import rbm
import regulation_set
from theano.tensor.shared_randomstreams import RandomStreams
import pandas
import os
import timeit
import regulation_set as rs
from sklearn.cross_validation import train_test_split
from optparse import OptionParser

def makeMatricesAgree(Mat1,Mat1Label,Mat2,Mat2Label):
    df1=pandas.DataFrame(Mat1,columns=Mat1Label)
    df2=pandas.DataFrame(Mat2,columns=Mat2Label)

    intersection=list(set(Mat1Label) & set(Mat2Label))

    out1=df1[intersection].as_matrix()
    out2=df2[intersection].as_matrix()
    return(out1,out2,intersection)

class masked_GBRBM(rbm.RBM):
    def __init__(
        self,
        input,
        n_visible,
        n_hidden,
        W=None,
        M=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param M: a binary mask matrix, same size as W, that indicates where a particular connection can be used

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if M is None:
            M=numpy.ones(shape=(n_visible, n_hidden))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            #initial_0=numpy.zeros(shape=(n_visible, n_hidden))
            #Carlin: multiply by the mask to zero out the correct entries
            initial_W=numpy.multiply(initial_W,M)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.initial_W=W.get_value()
        M = theano.shared(value=M, name='M', borrow=True)
        a=theano.tensor.dmatrix()
        b=theano.tensor.dmatrix()
        self.eleMult = theano.function([a, b], a * b)
        self.W = W
        self.M= M
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):

        #changed for gaussian visible units

        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term + vbias_term

    def sample_v_given_h(self, h0_sample):

        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''
        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead
        #v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        v1_sample = pre_sigmoid_v1
        #Carlin: because the connection matrix is sparse, we want reintroduce some stochasticity here

        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
        containing old state of Gibbs chain. This must be a shared
        variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        #apply the mask to the gradient

        gparams[0]=gparams[0]*self.M.get_value()

        #print gparams[0].get_value(borrow=True)

        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        n=0
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            if n==0:
                updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
                n=n+1
            else:
                updates[param] = param - gparam * T.cast(
                    lr,
                    dtype=theano.config.floatX
                )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

def train_masked_rbm(train_set_x_full_df,
                    regulators,learning_rate=0.03, training_epochs=500, batch_size=20,
                    output_folder='rbm_output/'
                    ):
    #first load the sets data


    #then load the data
    #train_set_x is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example.

    entities_x=list(train_set_x_full_df.columns.values)[1:]
    sample_names=list(train_set_x_full_df[train_set_x_full_df.columns[0]])

    [regulators_agree,train_set_x_full,entities_agree]=makeMatricesAgree(regulators.interaction_matrix.toarray(),regulators.entities_order,train_set_x_full_df,entities_x)

    train_set_x_full = theano.shared(numpy.asarray(train_set_x_full,
                                               dtype=theano.config.floatX),
                                 borrow=True)

    regulators_agree = theano.shared(numpy.asarray(regulators_agree.T,
                                               dtype=theano.config.floatX),
                                 borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x_full.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, regulators_agree.get_value(borrow=True).shape[1]),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    masked_rbm = masked_GBRBM(input=x, n_visible=regulators_agree.get_value(borrow=True).shape[0],
    n_hidden=regulators_agree.get_value(borrow=True).shape[1], numpy_rng=rng, theano_rng=theano_rng,M=regulators_agree.get_value(borrow=True))

    # get the cost and the gradient corresponding to one step of CD-15
    #cost, updates = masked_rbm.get_cost_updates(lr=learning_rate,
    #                                    persistent=persistent_chain, k=15)

    cost, updates = masked_rbm.get_cost_updates(lr=learning_rate,
                                         persistent=None, k=15)

    #################################
    #     Training the RBM          #
    #################################

    #################################
    #     Training the RBM          #
    #################################
    #if not os.path.isdir(output_folder):
    #    os.makedirs(output_folder)
    #os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x_full[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #h=masked_rbm.propup(train_set_x_full.get_value())
    pre_sigmoid_ph, ph_mean, ph_sample = masked_rbm.sample_h_given_v(train_set_x_full)

    return masked_rbm,ph_mean,regulators.regulators_order,sample_names,entities_agree

def cross_train_masked_RBM(x_full_df,
                           regulators,n_fold=5):

    #assign a fold to each row

    for i in xrange(n_fold):
        X_train, X_test = train_test_split(x_full_df,test_size=0.2)


if __name__ == '__main__':
    #parse input options
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="train_data_file", action="store", type="string", default='/Users/danielcarlin/projects/regulator_RBM/test_data/all_data.tab',
                      help="File containining a samples (rows) by genes (columns), tab delimited data")
    parser.add_option("-r", "--regulators", dest="regulator_list_file", action="store", type="string", default='/Users/danielcarlin/Data/Multinet/regulatory_filtlered5.list_t',
                      help="File containining a list of lists (in list_t format), first column is the list name, subsequent columns are genes associated with that list")
    parser.add_option('-o', "--output", dest="output", action="store", type="string", default='output.txt',help ="output file of hidden layer probabilities")
    parser.add_option('-l',"--learning-rate", dest="learning_rate", action='store', type='float', default=0.03, help="learning rate for RBM, default=0.03.  Generally, if things aren't converging, decrease this value")
    parser.add_option('-m', "--connection-matrix", dest="c_mat_file", action='store', type='string', default=None, help="output connection matrix to file")
    parser.add_option('-c', "--correlation-file", dest="corr_file", action='store', type='string', default=None, help="file for correlation between expression and regulon")
    xval=False
    (opts, args) = parser.parse_args()

    train_set_x_full_df = pandas.read_table(opts.train_data_file)
    regulators=rs.regulator_list(opts.regulator_list_file)
    #opts.c_mat_file='/Users/danielcarlin/PycharmProjects/DomainRegulation/conn_out.tab'

    if xval:
        cross_train_masked_RBM(x_full_df=train_set_x_full_df,regulators=regulators)
    else:
        mRBM,train_mean_h,regulons,sample_names,entities=train_masked_rbm(train_set_x_full_df=train_set_x_full_df,regulators=regulators,learning_rate=opts.learning_rate)
        h_prob=train_mean_h.eval()
        out_df=pandas.DataFrame(h_prob,columns=regulons,index=sample_names)
        out_df.to_csv(path_or_buf=opts.output,sep='\t')

    if opts.c_mat_file is not None:
        outConn=pandas.DataFrame(mRBM.W.eval(),columns=regulons, index=entities)
        outConn.to_csv(path_or_buf=opts.c_mat_file, sep='\t')
