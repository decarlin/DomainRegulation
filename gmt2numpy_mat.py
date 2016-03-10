__author__ = 'danielcarlin'

from optparse import OptionParser
import numpy as np
from array import array
from scipy.sparse import coo_matrix

def readgmt(filename):
    sets={}
    all_members=set()
    with open(filename,'r') as f:
        arr = f.readline().rstrip().split('\t')
        sets[arr[0]]=arr[2:(length(arr)-1)]
        all_members.add(sets[arr[0]])
    return list(sets,all_members)

def lists2binaryMat(sets,members):
    outMat=np.


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="infile", action="store", type="string", help="Input File")
    parser.add_option("-o", "--output", dest="infile", action="store", type="string", help="Output File")

