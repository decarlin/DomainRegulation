#!/usr/bin/env python

from optparse import OptionParser

def parseBoundaries(infile):
    boundaries=[]
    with open(infile, "r") as ins:
            for line in ins:
                arr=line.split('\t')
                boundaries.append((arr[0],arr[1]))
    return boundaries

def sortedBoundaries2domainBed(LofTuples):
    start_chr = True
    domains = []
    curr_coord=None
    curr_chr=None
    for (chr,coord) in LofTuples:
        if start_chr:
            curr_coord=coord
            curr_chr=chr
            last_coord=0
            start_chr=False
        elif chr==curr_chr:
            last_coord=curr_coord
            curr_coord=coord
        else:
            start_chr=True
            continue

        name="_".join([chr,str(last_coord),str(curr_coord)])
        domains.append((chr,str(last_coord),str(curr_coord),name))
    return domains

if __name__ == "__main__":

    parser=OptionParser()
    parser.add_option("-i","--input", dest="infile", action="store", type="string", help="Input File")
    parser.add_option("-o","--output", dest="outfile", action="store", type="string", help="Output File")

    (opts,args)=parser.parse_args()

    boundaries=parseBoundaries(opts.infile)
    domains=sortedBoundaries2domainBed(boundaries)

    out=open(opts.outfile,'w')
    for row in domains:
        out.write("\t".join(row)+"\n")
