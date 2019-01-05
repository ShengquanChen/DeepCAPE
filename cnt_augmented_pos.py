#!/usr/bin/env python
# encoding: utf-8
import os
import hickle as hkl
import numpy as np

def main(name, l, stride, ratio):
    """
    Count the number of augmented samples for each raw enhancer.
    Prepare for the training data of pre-training stage in 'train_DeepCAPE.py'.
    """
    for i in range(5):
        TRAIN_POS_FA = './data1_%d/res_%s_stride%d/%s_train_stride%d_positive_%d.bed'% (ratio, name, stride, name, stride, i)
        fin = open(TRAIN_POS_FA, 'r')
        entries = fin.readlines()
        fin.close()
        pos_id = list()
        for line in entries:
            pos_id.append(line.split('\t')[3])
        raw_enhancers = sorted(set(pos_id),key=pos_id.index)
        augmented_num = [pos_id.count(item) for item in raw_enhancers]
        print('Totally %d raw enhancers and %d augmented_pos_num' % (len(raw_enhancers), sum(augmented_num)))
        hkl.dump(augmented_num, './augmented_pos_num/ratio%d_%s_train_stride%d_%d.hkl' % (ratio, name, stride, i), 'w')

if __name__ == "__main__":
    names = ['epithelial_cell_of_esophagus','melanocyte','cardiac_fibroblast','keratinocyte','myoblast','stromal','mesenchymal','natural_killer','monocyte']
    print names
    name   = raw_input('Choose tissue : ')
    stride = input('Choose stride: ')
    main(name, 300, stride, 10)
    main(name, 300, stride, 20)