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
import string

# Genome sequence chr1-22,X,Y
GENOME = './Chromosomes/'
INDEX1 = './temp/res_complement.bed'    # Genome except the following regions
ENHANCER    = './temp/enhancer.txt'     # 800k known enhancers data
PROMOTER    = './temp/promoter.txt'     # 33k  known promoters data
LNCPROMOTER = './temp/lncpromoter.txt'  # 153k known lncpromoters data
LNCRNA      = './temp/lncrna.txt'       # 89k  known lncrnas data
EXON        = './temp/genes.gtf.gz'     # 963k known exons(CDS) data

all_split_neg = {}

acgt2num = {'A': 0,
            'C': 1,
            'G': 2,
            'T': 3}
complement = {'A': 'T',
              'C': 'G',
              'G': 'C',
              'T': 'A'}

name2index = {'res_complement':INDEX1}

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
    print 'Find corresponding hkl file'
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

def checkseq(chrkey, start, end):
    sequence = sequences[chrkey][start:end]
    legal = ('n' not in sequence) and ('N' not in sequence)
    return sequence, legal

def chunks(l, n, o):
    """
    Yield successive n-sized chunks with o-sized overlap from l.
    """
    return [l[i: i + n] for i in range(0, len(l), n-o)]

def cropseq(l, bedfile):
    """
    Generate negative samples with specific length.
    """
    print 'Generating pool_neg samples with length {} bps...'.format(l)
    stride = l
    fin = open(INDEX1, 'r').readlines()
    neg_samples = 0
    f = open(bedfile, 'w')
    for index in fin:
        chrkey, startpos, endpos = index.strip().split('\t')
        startpos = int(startpos)
        endpos = int(endpos)
        l_orig = endpos - startpos
        if l_orig >= l:
            chunks_ = chunks(range(startpos, endpos), l, l - stride)
            for chunk in chunks_:
                start = chunk[0]
                end = chunk[-1] + 1
                if (end - start) == l:
                    seq, legal = checkseq(chrkey, start, end)
                    if legal:
                        seq = seq.upper()
                        gc_content = 1.0 * (seq.count('G') + seq.count('C')) / l
                        f.write('%s\t%d\t%d\t%.3f\t.\t.\n' % (chrkey, start, end, gc_content))
                        neg_samples += 1
                elif (end - start) < l:
                    break
    print 'Generated {} pool_neg samples.'.format(neg_samples)
    return neg_samples

def main(name, ind):
    """
    Generate negative samples with different lengths of raw enhancers.
    Prepare for the generation of chunked negative samples in 'gen_seq_data.py'.
    """
    length = hkl.load('./pool_neg/neg_len/%d.hkl' % ind) # different lengths of raw enhancers (saved in 10 files)
    print 'Generating pool_neg of %d different length ...' % length.shape[0]
    for l in length:
        filename = './pool_neg/length_%d.bed'% l
        neg_samples = cropseq(l, filename)

if __name__ == "__main__":
    name   = 'res_complement'
    ind = input("Choose index (1-10): ")
    main(name, ind)