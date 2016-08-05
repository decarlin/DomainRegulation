__author__ = 'danielcarlin'

import regulation_set as rs
filename='/Users/danielcarlin/Data/Multinet/multinet_regulons.list_t'
out_filename='/Users/danielcarlin/Data/Multinet/multinet_coregulation.sif'


fh = open(filename)

interaction_list=list()
coregulation_list=list()

for file_line in fh:
    line=file_line.rstrip('\n').split('\t')
    local_entities=list()
    for i in range(1,len(line)):
            interaction_list.append((line[0],line[i]))
            local_entities.append(line[i])
            if i>1:
                for j in range(0,i-1):
                    coregulation_list.append((line[i],'coreg_'+line[0],local_entities[j]))

fh.close()

outfh=open(out_filename,'w')

for int in interaction_list:
    outfh.write(int[0]+'\tregulates\t'+int[1]+'\n')

for coreg in coregulation_list:
    outfh.write('\t'.join(coreg)+'\n')