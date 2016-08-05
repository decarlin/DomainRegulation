#!/usr/bin/env python

__author__ = 'danielcarlin'
import pandas
import scipy.stats
import numpy.random
from optparse import OptionParser
from theano_maskedRBM import makeMatricesAgree

def corr_matrices(data_in,rbm_out,spearman=True):
    """Take two matrices and return the correlation of their corresponding columns. Defaults to Spearman."""
    entities_x=list(data_in.columns.values)[1:]
    sample_names=list(data_in[data_in.columns[0]])

    entities_out=list(rbm_out.columns.values)[1:]
    out_names=list(rbm_out[rbm_out.columns[0]])

    [m1,m2,entities_agree]=makeMatricesAgree(rbm_out,entities_out,data_in,entities_x)

    spr={}
    null_model={}

    for i in xrange(m1.shape[1]):
        if spearman:
            spr[i]=scipy.stats.spearmanr(m1[:,i],m2[:,i])
            null_model[i]=scipy.stats.spearmanr(m1[:,i],m2[:,numpy.random.randint(low=0,high=m2.shape[1])])
        else:
            spr[i]=scipy.stats.pearsonr(m1[:,i],m2[:,i])
            null_model[i]=scipy.stats.pearsonr(m1[:,i],m2[:,numpy.random.randint(low=0,high=m2.shape[1])])

    return spr,entities_agree, null_model



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="train_data_file", action="store", type="string", default='/Users/danielcarlin/projects/regulator_RBM/test_data/all_data.tab',
                      help="File containining a samples (rows) by genes (columns), tab delimited data")
    parser.add_option('-r', "--rbm-output", dest="rbm_output_file", action="store", type="string", default='output.txt',help ="output file of hidden layer probabilities")
    parser.add_option('-c', "--correlation-file", dest="corr_file", action='store', type='string', default=None, help="file for correlation between expression and regulon")
    parser.add_option('-n', "--null-model", dest="null_file", action='store', type='string', default=None, help="file for null model output")
    parser.add_option('-p', "--pearson", dest="pearson", action='store_true', default=False, help="Pearson rather than Spearman correlation")

    (opts, args) = parser.parse_args()

    data_in = pandas.read_table(opts.train_data_file)
    rbm_out = pandas.read_table(opts.rbm_output_file)

    if opts.pearson:
        spr,ent,nm=corr_matrices(data_in,rbm_out,spearman=False)
    else:
        spr,ent,nm=corr_matrices(data_in,rbm_out)

    fh=open(opts.corr_file,'w')

    for k in spr.keys():
        fh.write(ent[k]+'\t'+str(spr[k][0])+'\t'+str(spr[k][1])+'\n')

    fh.close()

    fh2=open(opts.null_file,'w')

    for k in spr.keys():
        fh2.write(ent[k]+'\t'+str(nm[k][0])+'\t'+str(nm[k][1])+'\n')

    fh2.close()