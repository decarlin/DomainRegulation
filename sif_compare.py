__author__ = 'danielcarlin'

import scipy.stats as stats

def parseNet(network):
    """
    Parse .sif network, using just the first and third columns
    to build an undirected graph. Store the node out-degrees
    in an index while we're at it.
    """
    edges = set()
    nodes = set()
    degrees = {}
    for line in open(network, 'r'):

        parts = line.rstrip().split("\t")

        if len(parts) != 3:
            print("problem line:"+line)
            continue

        source = parts[0]
        target = parts[2]

        # if inputing a multi-graph, skip this
        if (source, target) in edges:
            continue
        if source==target:
            continue

        edges.add((source, target))
        edges.add((target, source))
        nodes.add(source)
        nodes.add(target)

        if source not in degrees:
            degrees[source] = 0
        if target not in degrees:
            degrees[target] = 0

        degrees[source] += 1
        degrees[target] += 1

    return (edges, nodes, degrees)

def CountEdgeOverlap(net1,net2):
    overlapcounter=0
    for edge1 in net1:
        if (edge1 in net2):
            overlapcounter=overlapcounter+1

    return(overlapcounter)

file1='/Users/danielcarlin/Data/Multinet/multinet_coregulation.sif'
file2='/Users/danielcarlin/Data/CellNet/CellNet_all.sif'

(edge1,nodes1,degrees1)=parseNet(file1)
(edge2,nodes2,degrees2)=parseNet(file2)

N=CountEdgeOverlap(edge1,edge2)
pval=1-stats.hypergeom(len(edge1)+len(edge2),len(edge1),len(edge2)).sf(N)


print 'Overlap:'+str(N)
print 'Size 1:'+str(len(edge1))
print 'Size 2:'+str(len(edge2))
print 'One-tailed pval:'+pval
