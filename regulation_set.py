from scipy.sparse import coo_matrix
from array import array

class regulator_list:
    def __init__(self,filename):
        fh = open(filename)
        interaction_list=list()
        entities=set()
        regulators=set()

        regulators2index={}
        index2regulators={}
        entities2index={}
        index2entities={}


        for file_line in fh:
            line=file_line.rstrip('\n').split('\t')
            regulators.add(line[0])
            for i in range(1,len(line)):
                entities.add(line[i])
                interaction_list.append((line[0],line[i]))


        entities_order = list(entities)
        for i in range(0, len(entities)):
            index2entities[i] = entities_order[i]
            entities2index[entities_order[i]] = i

        regulators_order = list(regulators)
        for j in range(0,len(regulators)):
            index2regulators[j] = regulators_order[j]
            regulators2index[regulators_order[j]]=j

        # SCIPY uses row and column indexes to build the matrix
        # row and columns are just indexes: the data column stores
        # the actual entries of the matrix

        row = array('i')
        col = array('i')
        data = array('i')

        for interaction in interaction_list:
            # append index to i-th row, j-th column
            row.insert(len(row), regulators2index[interaction[0]])
            col.insert(len(col), entities2index[interaction[1]])
            # 1 for  adjacency matrix
            data.insert(len(data), 1)

        # Build the graph laplacian: the CSC matrix provides a sparse matrix format
        # that can be exponentiated efficiently
        self.interaction_matrix = coo_matrix((data,(row, col)), shape=(len(regulators),len(entities))).tocsc()
        self.interaction_list=interaction_list
        self.regulators_order=regulators_order
        self.entities_order=entities_order