#!/usr/bin/env python
# encoding: utf-8
import gzip
import sys
import cPickle as pkl
import hickle as hkl
import numpy as np
import bisect
import random
import os
import copy
from sklearn.cross_validation import KFold

# Genome sequence chr1-22,X,Y
GENOME = './Chromosomes/'
INDEX1 = './temp/specific/monocyte.bed'
INDEX2 = './temp/specific/mesenchymal.bed'
INDEX3 = './temp/specific/keratinocyte.bed'
INDEX4 = './temp/specific/myoblast.bed'
INDEX5 = './temp/specific/melanocyte.bed'
INDEX6 = './temp/specific/stromal.bed'
INDEX7 = './temp/specific/natural_killer.bed'
INDEX8 = './temp/specific/cardiac_fibroblast.bed'
INDEX9 = './temp/specific/epithelial_cell_of_esophagus.bed'


acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}
complement = {'A': 'T',
              'C': 'G',
              'G': 'C',
              'T': 'A'}

name2index = {'monocyte': INDEX1,
              'mesenchymal':INDEX2,
              'keratinocyte':INDEX3,
              'myoblast':INDEX4,
              'melanocyte':INDEX5,
              'stromal':INDEX6,
              'natural_killer':INDEX7,
              'cardiac_fibroblast':INDEX8,
              'epithelial_cell_of_esophagus':INDEX9}

def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)
    mat = np.zeros((h, w), dtype=bool)  # True or False in mat
    for i in xrange(w):
        mat[acgt2num[seq[i]], i] = 1.
    return mat.reshape((1, -1))

# Load whole genome sequence
print 'Loading whole genome sequence...'
chrs = range(1, 23)
chrs.extend(['X', 'Y'])
keys = ['chr' + str(x) for x in chrs]
if os.path.isfile('./temp/sequences.hkl'):
    sequences = hkl.load('./temp/sequences.hkl')
else:
    sequences = dict()
    for i in range(24):
        fa = open('%s%s.fa' % (GENOME, keys[i]), 'r')
        sequence = fa.read().splitlines()[1:]
        fa.close()
        sequence = ''.join(sequence)
        sequences[keys[i]]= sequence
    hkl.dump(sequences, './temp/sequences.hkl', 'w')

all_neg = dict()
for key in keys:
    all_neg[key] = list()

def checkseq(chrkey, start, end):
    sequence = sequences[chrkey][start:end]
    legal = ('n' not in sequence) and ('N' not in sequence)
    return sequence, legal

def loadindex(name, ratio):
    """
    Load enhancers indexes(id, chr, start, end).
    """
    print 'Loading %s enhancer indexes...' % name
    if os.path.isfile('./temp/specific/%s_index_ratio%d.hkl' % (name, ratio)):
       print 'Find corresponding hkl file'
       indexes, neg_indexes = hkl.load('./temp/specific/%s_index_ratio%d.hkl' % (name, ratio))
       return indexes, neg_indexes
    f = open(name2index[name], 'r')
    entries = f.readlines()
    f.close()
    indexes = list()
    neg_indexes = list()

    lens = list()
    for i, entry in enumerate(entries):
        chrkey, start, end = entry.split(' ')
        lens.append(int(end) - int(start))
    lens = np.array(lens)
    bot = np.percentile(lens, 5)
    top = np.percentile(lens, 95)
    for i, entry in enumerate(entries):
        chrkey, start, end = entry.split(' ')
        start = int(start) - 1
        end = int(end) - 1
        seq, legal = checkseq(chrkey, start, end)
        length = end - start
        if legal and length > bot and length < top:
            indexes.append(['%s%05d' % (name, i), chrkey, start, end])
            seq = seq.upper()
            pos_cont = 1.0 * (seq.count('G') + seq.count('C')) / length
            f = open('./pool_neg/length_%d.bed' % length, 'r')
            neg_entries = f.readlines()
            f.close()
            # random.shuffle(neg_entries)
            abs_cont = list()
            for item in neg_entries:
                neg_cont = item.split('\t')[3]
                abs_cont.append(abs(pos_cont - float(neg_cont)))
            sorted_ind = np.argsort(np.array(abs_cont)).tolist()

            cnt = 0
            for ind in sorted_ind:
                neg_chrkey, neg_start, neg_end = neg_entries[ind].split('\t')[:3]
                neg_start = int(neg_start) - 1
                neg_end = int(neg_end) - 1
                flag = 1
                for item in all_neg[neg_chrkey]:
                    if (neg_start >= item[0] and neg_start <= item[1]) or (neg_end >= item[0] and neg_end <= item[1]) or (neg_end >= item[0] and neg_start <= item[0]) or (neg_end >= item[1] and neg_start <= item[1]) :
                        flag = 0
                        break
                if flag:
                    cnt += 1
                    all_neg[neg_chrkey].append([neg_start, neg_end])
                    neg_indexes.append(['neg%05d_%d' % (i, cnt), neg_chrkey, neg_start, neg_end])
                    if cnt == ratio:
                        break
            if cnt != ratio:
                print '[Error! No enough negative samples!]'

    print 'Totally {0} enhancers and {1} negative samples.'.format(len(indexes), len(neg_indexes))
    hkl.dump([indexes, neg_indexes], './temp/specific/%s_index_ratio%d.hkl' % (name, ratio), 'w')
    return [indexes, neg_indexes]

def chunks(l, n, o):
    """
    Yield successive n-sized chunks with o-sized overlap from l.
    """
    return [l[i: i + n] for i in range(0, len(l), n-o)]

def train_test_split(indexes, neg_indexes, name, ratio):
    """
    Split training and test sets.
    (indexes: indexes to be split)
    """
    print 'Splitting the indexes into train and test parts...'
    file_name = './temp/specific/%s_index_split_ratio%d.hkl' % (name, ratio)
    if os.path.isfile(file_name):
       print 'Loading saved splitted indexes...'
       return hkl.load(file_name)
    n_samles = len(indexes)
    kfs = KFold(n_samles, n_folds=5, shuffle=True)
    allsplits = []
    neg_allsplits =[]
    for train, test in kfs:
        train_indexes = [indexes[i] for i in train]
        test_indexes = [indexes[i] for i in test]
        allsplits.append((train_indexes, test_indexes))
        neg_train =[]
        for i in train:
            neg_train = neg_train + range(i*ratio, (i+1)*ratio)
        neg_test =[]
        for i in test:
            neg_test = neg_test + range(i*ratio, (i+1)*ratio)
        neg_train_indexes = [neg_indexes[i] for i in neg_train]
        neg_test_indexes = [neg_indexes[i] for i in neg_test]
        neg_allsplits.append((neg_train_indexes, neg_test_indexes))
    print 'Saving splitted indexes...'
    hkl.dump([allsplits, neg_allsplits], file_name)
    return [allsplits, neg_allsplits]

def to_bed_file(indexes, bedfile):
    """
    Format indexes into bed file.
    (indexes: indexes to be saved to bed files. bedfile: destination bed file name)
    """
    print 'Saving sequences into %s...' % bedfile
    f = open(bedfile, 'w')
    for index in indexes:
        if len(index) == 4:
            [id, chrkey, start, end] = index
            f.write('{0[1]}\t{0[2]}\t{0[3]}\t{0[0]}\t.\t.\n'.format(index))
        elif len(index) == 5:
            [id, chrkey, start, end, seq] = index
            f.write('{0[1]}\t{0[2]}\t{0[3]}\t{0[0]}\t.\t.\n'.format(index))
        else:
            raise ValueError('index not in correct format!')
    f.close()

def cropseq(indexes, l, stride):
    """
    Generate chunked samples according to loaded index.
    """
    enhancers = list()
    for index in indexes:
        try:
            [sampleid, chrkey, startpos, endpos, _] = index
        except:
            [sampleid, chrkey, startpos, endpos] = index
        l_orig = endpos - startpos
        if l_orig < l:
            for shift in range(0, l - l_orig, stride):
                start = startpos - shift
                end = start + l
                seq, legal = checkseq(chrkey, start, end)
                if legal and len(seq) == l:
                    enhancers.append((sampleid, chrkey, start, end, seq))
        elif l_orig >= l:
            chunks_ = chunks(range(startpos, endpos), l, l - stride)
            for chunk in chunks_:
                start = chunk[0]
                end = chunk[-1] + 1
                if (end - start) == l:
                    seq, legal = checkseq(chrkey, start, end)
                    if legal and len(seq) == l:
                        enhancers.append((sampleid, chrkey, start, end, seq))
                else:
                    break
    print 'Data augmentation: from {} indexes to {} samples'.format(len(indexes), len(enhancers))
    return enhancers

def main(name, l, stride, ratio):
    indexes, neg_indexes = loadindex(name, ratio)
    allsplits, neg_allsplits = train_test_split(indexes, neg_indexes , name, ratio)
    for i in range(5):
        train_indexes, test_indexes = allsplits[i]
        neg_train_indexes, neg_test_indexes = neg_allsplits[i]

        TRAIN_POS_FA = './data1_%d/%s_train_stride%d_positive_%d.bed'% (ratio, name, stride, i)
        TEST_POS_FA  = './data1_%d/%s_test_stride%d_positive_%d.bed' % (ratio, name, stride, i)
        TRAIN_NEG_FA = './data1_%d/%s_train_stride%d_negative_%d.bed'% (ratio, name, stride, i)
        TEST_NEG_FA  = './data1_%d/%s_test_stride%d_negative_%d.bed' % (ratio, name, stride, i)

        train_pos = cropseq(train_indexes, l, stride)
        test_pos  = cropseq(test_indexes, l, stride)
        to_bed_file(train_pos, TRAIN_POS_FA)
        to_bed_file(test_pos, TEST_POS_FA)

        train_neg = cropseq(neg_train_indexes, l, stride)
        test_neg  = cropseq(neg_test_indexes, l, stride)
        to_bed_file(train_neg, TRAIN_NEG_FA)
        to_bed_file(test_neg, TEST_NEG_FA)

        train_y = [1] * len(train_pos) + [0] * len(train_neg)
        train_y = np.array(train_y, dtype=bool)
        train_X_pos = np.vstack([seq2mat(item[-1]) for item in train_pos])
        train_X_neg = np.vstack([seq2mat(item[-1]) for item in train_neg])
        train_X = np.vstack((train_X_pos, train_X_neg))

        test_y = [1] * len(test_pos) + [0] * len(test_neg)
        test_y = np.array(test_y, dtype=bool)
        test_pos_ids = [item[0] for item in test_pos]
        test_neg_ids = [item[0] for item in test_neg]
        test_X_pos = np.vstack([seq2mat(item[-1]) for item in test_pos])
        test_X_neg = np.vstack([seq2mat(item[-1]) for item in test_neg])
        test_X = np.vstack((test_X_pos, test_X_neg))
        hkl.dump((train_X, train_y), './data1_%d/%s_train_stride%d_%d_seq.hkl' % (ratio, name, stride, i), 'w')
        hkl.dump((test_X, test_y), './data1_%d/%s_test_stride%d_%d_seq.hkl' % (ratio, name, stride, i), 'w')
        hkl.dump(test_pos_ids, './data1_%d/%s_test_pos_ids_stride%d_%d.hkl' % (ratio, name, stride, i), 'w')
        hkl.dump(test_neg_ids, './data1_%d/%s_test_neg_ids_stride%d_%d.hkl' % (ratio, name, stride, i), 'w')

if __name__ == "__main__":
    names = ['epithelial_cell_of_esophagus','melanocyte','cardiac_fibroblast','keratinocyte','myoblast','stromal','mesenchymal','natural_killer','monocyte']
    print names
    name   = raw_input('Choose tissue : ')
    stride = input('Choose stride: ')
    #ratio  = input('n_pos:n_neg=1: ')
    main(name, 300, stride, 10)
    main(name, 300, stride, 20)