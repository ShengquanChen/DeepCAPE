#!/usr/bin/env python
# encoding: utf-8
import os
gpu_use = raw_input('Use gpu: ')
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
import tensorflow as tf
import sys
import numpy as np
from random import shuffle
import random
import hickle as hkl
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, LSTM, Bidirectional, Permute, Reshape, Merge, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations

def load_dataset(model_ind, ratio, name, dataset_ind, stride):
    # Determine whether to carry out pre-training.
    pretrain_flag = 0
    filename = './results1_%d/pretrain_%s_stride%dmodel%ddataset%d.h5' % (ratio, name, stride, model_ind, dataset_ind)
    if not os.path.isfile(filename):
        pretrain_flag = 1
        print '[===Pretrain===]'
    # Load sequence data.
    dataset = './data1_%d/%s_train_stride%d_%d_seq.hkl'%(ratio, name, stride, dataset_ind)
    X_train_seq, y_train = hkl.load(dataset)
    X_train_seq = X_train_seq.reshape(-1, 4, 300, 1)
    y_train = np.array(y_train, dtype='uint8')
    print('[X_train_seq shape: {}, y_train shape: {}]'.format(X_train_seq.shape, y_train.shape))
    # Load chromatin accessibility data produced by the auto-encoder module.
    dataset = './data1_%d/%s_train_stride%d_%d_dnase_AE.hkl'%(ratio, name, stride, dataset_ind)
    X_train_dnase, y_train = hkl.load(dataset)
    X_train_dnase = X_train_dnase.reshape(-1, 1, 300, 1)
    y_train = np.array(y_train, dtype='uint8')
    print('[X_train_dnase shape: {}, y_train shape: {}]'.format(X_train_dnase.shape, y_train.shape))

    if pretrain_flag == 1:
        # Extract the same number of positive and negative samples for pre-training.
        augmented_pos_num = hkl.load('./augmented_pos_num/ratio%d_%s_train_stride%d_%d.hkl' % (ratio, name, stride, dataset_ind))
        pretrain_index = list()
        for i in range(X_train_dnase.shape[0]):
            if y_train[i] == 1:
                pretrain_index.append(i)
        neg_start_index = len(pretrain_index)
        for item in augmented_pos_num:
            pretrain_index += range(neg_start_index, neg_start_index+item)
            neg_start_index = neg_start_index + item*ratio
        X_train_seq = X_train_seq[pretrain_index]
        X_train_dnase = X_train_dnase[pretrain_index]
        y_train = y_train[pretrain_index]
    # Shuffle the training samples
    indice = np.arange(X_train_dnase.shape[0])
    np.random.shuffle(indice)
    X_train_seq = X_train_seq[indice]
    X_train_dnase = X_train_dnase[indice]
    y_train = y_train[indice]

    return X_train_seq, X_train_dnase, y_train


def model_training(model_ind, ratio, name, dataset_ind, stride, X_train_seq, X_train_dnase, y_train):
    # Determine whether to carry out pre-training.
    pretrain_flag = 0
    filename = './results1_%d/pretrain_%s_stride%dmodel%ddataset%d.h5' % (ratio, name, stride, model_ind, dataset_ind)
    if not os.path.isfile(filename):
        pretrain_flag = 1
        print '[===Pretrain===]'

    if model_ind == 6:# DNA module only
        input_seq  = Input(shape=(4, 300, 1))
        seq_conv1_ = Convolution2D(128, 4, 8, activation='relu',border_mode='valid',dim_ordering='tf')
        seq_conv1  = seq_conv1_(input_seq)
        seq_conv2_ = Convolution2D(64, 1, 1, activation='relu',border_mode='same')
        seq_conv2  = seq_conv2_(seq_conv1)
        seq_conv3_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv3  = seq_conv3_(seq_conv2)
        seq_conv4_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        seq_conv4  = seq_conv4_(seq_conv3)
        seq_pool1  = MaxPooling2D(pool_size=(1, 2))(seq_conv4)
        seq_conv5_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv5  = seq_conv5_(seq_pool1)
        seq_conv6_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv6  = seq_conv6_(seq_conv5)
        #
        seq_conv7_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        seq_conv7  = seq_conv7_(seq_conv6)
        #
        seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv7)
        merge_seq_conv2_conv3 = merge([seq_conv2, seq_conv3], mode = 'concat', concat_axis = -1)
        merge_seq_conv5_conv6 = merge([seq_conv5, seq_conv6], mode = 'concat', concat_axis = -1)
        x = merge([seq_conv1, merge_seq_conv2_conv3, merge_seq_conv5_conv6, seq_pool2], mode = 'concat', concat_axis = 2)
        x = Flatten()(x)
        dense1_ = Dense(512, activation='relu')
        dense1  = dense1_(x)
        dense2  = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3 = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[input_seq], output=[pred_output])

    if model_ind == 7:# DNase module only
        input_dnase  = Input(shape=(1, 300, 1))
        dnase_conv1_ = Convolution2D(128, 1, 8, activation='relu',border_mode='valid',dim_ordering='tf')
        dnase_conv1  = dnase_conv1_(input_dnase)
        dnase_conv2_ = Convolution2D(64, 1, 1, activation='relu',border_mode='same')
        dnase_conv2  = dnase_conv2_(dnase_conv1)
        dnase_conv3_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv3  = dnase_conv3_(dnase_conv2)
        dnase_conv4_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        dnase_conv4  = dnase_conv4_(dnase_conv3)
        dnase_pool1  = MaxPooling2D(pool_size=(1, 2))(dnase_conv4)
        dnase_conv5_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv5  = dnase_conv5_(dnase_pool1)
        dnase_conv6_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv6  = dnase_conv6_(dnase_conv5)
        #
        dnase_conv7_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        dnase_conv7  = dnase_conv7_(dnase_conv6)
        #
        dnase_pool2  = MaxPooling2D(pool_size=(1, 2))(dnase_conv7)
        merge_dnase_conv2_conv3 = merge([dnase_conv2, dnase_conv3], mode = 'concat', concat_axis = -1)
        merge_dnase_conv5_conv6 = merge([dnase_conv5, dnase_conv6], mode = 'concat', concat_axis = -1)
        x = merge([dnase_conv1, merge_dnase_conv2_conv3, merge_dnase_conv5_conv6, dnase_pool2], mode = 'concat', concat_axis = 2)
        x = Flatten()(x)
        dense1_ = Dense(512, activation='relu')
        dense1  = dense1_(x)
        dense2  = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3 = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[input_dnase], output=[pred_output])

    if model_ind == 8:# DeepCAPE
        input_seq   = Input(shape=(4, 300, 1))
        input_dnase = Input(shape=(1, 300, 1))
        seq_conv1_ = Convolution2D(128, 4, 8, activation='relu',border_mode='valid',dim_ordering='tf')
        seq_conv1  = seq_conv1_(input_seq)
        seq_conv2_ = Convolution2D(64, 1, 1, activation='relu',border_mode='same')
        seq_conv2  = seq_conv2_(seq_conv1)
        seq_conv3_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv3  = seq_conv3_(seq_conv2)
        seq_conv4_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        seq_conv4  = seq_conv4_(seq_conv3)
        seq_pool1  = MaxPooling2D(pool_size=(1, 2))(seq_conv4)
        seq_conv5_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv5  = seq_conv5_(seq_pool1)
        seq_conv6_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        seq_conv6  = seq_conv6_(seq_conv5)
        seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv6)
        dnase_conv1_ = Convolution2D(128, 1, 8, activation='relu',border_mode='valid',dim_ordering='tf')
        dnase_conv1  = dnase_conv1_(input_dnase)
        dnase_conv2_ = Convolution2D(64, 1, 1, activation='relu',border_mode='same')
        dnase_conv2  = dnase_conv2_(dnase_conv1)
        dnase_conv3_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv3  = dnase_conv3_(dnase_conv2)
        dnase_conv4_ = Convolution2D(128, 1, 1, activation='relu',border_mode='same')
        dnase_conv4  = dnase_conv4_(dnase_conv3)
        dnase_pool1  = MaxPooling2D(pool_size=(1, 2))(dnase_conv4)
        dnase_conv5_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv5  = dnase_conv5_(dnase_pool1)
        dnase_conv6_ = Convolution2D(64, 1, 3, activation='relu',border_mode='same')
        dnase_conv6  = dnase_conv6_(dnase_conv5)
        dnase_pool2  = MaxPooling2D(pool_size=(1, 2))(dnase_conv6)
        merge_seq_conv2_conv3 = merge([seq_conv2, seq_conv3], mode = 'concat', concat_axis = -1)
        merge_seq_conv5_conv6 = merge([seq_conv5, seq_conv6], mode = 'concat', concat_axis = -1)
        merge_dnase_conv2_conv3 = merge([dnase_conv2, dnase_conv3], mode = 'concat', concat_axis = -1)
        merge_dnase_conv5_conv6 = merge([dnase_conv5, dnase_conv6], mode = 'concat', concat_axis = -1)
        merge_pool2 = merge([seq_pool2, dnase_pool2], mode = 'concat', concat_axis = -1)
        x = merge([seq_conv1, merge_seq_conv2_conv3, merge_seq_conv5_conv6, merge_pool2, merge_dnase_conv5_conv6, merge_dnase_conv2_conv3, dnase_conv1], mode = 'concat', concat_axis = 2)
        x = Flatten()(x)
        dense1_ = Dense(512, activation='relu')
        dense1  = dense1_(x)
        dense2  = Dense(256, activation='relu')(dense1)
        x = Dropout(0.5)(dense2)
        dense3  = Dense(128, activation='relu')(x)
        pred_output = Dense(1, activation='sigmoid')(dense3)
        model = Model(input=[input_seq, input_dnase], output=[pred_output])

    if pretrain_flag == 1:
        # Carry out pre-training
        filename = './results1_%d/pretrain_%s_stride%dmodel%ddataset%d.h5' % (ratio, name, stride, model_ind, dataset_ind)
        early_stopping = EarlyStopping(monitor='val_loss', verbose=0, patience=3, mode='min')
        save_best = ModelCheckpoint(filename, save_best_only=True, save_weights_only=True)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', 'fbeta_score'])
        if model_ind == 6:
            model.fit(X_train_seq, y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])
        elif model_ind==7:
            model.fit(X_train_dnase, y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])
        elif model_ind==8:
            model.fit([X_train_seq, X_train_dnase], y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])

    else:
        if model_ind == 6:
            seq_conv1_.trainable = False
            seq_conv2_.trainable = False
            seq_conv3_.trainable = False
            seq_conv4_.trainable = False
            seq_conv5_.trainable = False
            seq_conv6_.trainable = False
            seq_conv7_.trainable = False
            dense1_.trainable    = False
            model = Model(input=[input_seq], output=[pred_output])
        elif model_ind == 7:
            dnase_conv1_.trainable = False
            dnase_conv2_.trainable = False
            dnase_conv3_.trainable = False
            dnase_conv4_.trainable = False
            dnase_conv5_.trainable = False
            dnase_conv6_.trainable = False
            dnase_conv7_.trainable = False
            dense1_.trainable      = False
            model = Model(input=[input_dnase], output=[pred_output])
        elif model_ind == 8:
            seq_conv1_.trainable = False
            seq_conv2_.trainable = False
            seq_conv3_.trainable = False
            seq_conv4_.trainable = False
            seq_conv5_.trainable = False
            seq_conv6_.trainable = False
            dnase_conv1_.trainable = False
            dnase_conv2_.trainable = False
            dnase_conv3_.trainable = False
            dnase_conv4_.trainable = False
            dnase_conv5_.trainable = False
            dnase_conv6_.trainable = False
            dense1_.trainable      = False
            model = Model(input=[input_seq, input_dnase], output=[pred_output])
        # Load model of pre-training.
        filename = './results1_%d/%s_stride%dmodel%ddataset%d.h5' % (ratio, name, stride, model_ind, dataset_ind)
        if not os.path.isfile(filename):
            filename = './results1_%d/pretrain_%s_stride%dmodel%ddataset%d.h5' % (ratio, name , stride, model_ind, dataset_ind)
        print '[Loading] ' + filename
        model.load_weights(filename)

        filename = './results1_%d/%s_stride%dmodel%ddataset%d.h5' % (ratio, name, stride, model_ind, dataset_ind)
        early_stopping = EarlyStopping(monitor='val_fbeta_score', verbose=0, patience=3, mode='max')
        save_best = ModelCheckpoint(filename, save_best_only=True, save_weights_only=True)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', 'fbeta_score'])
        if model_ind == 6:
            model.fit(X_train_seq, y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])
        elif model_ind == 7:
            model.fit(X_train_dnase, y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])
        elif model_ind == 8:
            model.fit([X_train_seq, X_train_dnase], y_train, batch_size=128, nb_epoch=30, validation_split=0.1, callbacks=[early_stopping, save_best])


if __name__ == "__main__":
    names = ['epithelial_cell_of_esophagus','melanocyte','cardiac_fibroblast','keratinocyte','myoblast','stromal','mesenchymal','natural_killer','monocyte']
    stride = input('Choose stride: ')
    model_ind   = input('Choose Model ([6]DNA module, [7]DNase module, [8]DeepCAPE) : ')
    for name in names:
        for dataset_ind in range(5):
            print "===========Training %s: %d===========" % (name, dataset_ind)
            for ratio in [10, 20]:
                X_train_seq, X_train_dnase, y_train = load_dataset(model_ind, ratio, name, dataset_ind, stride)
                model_training(model_ind, ratio, name, dataset_ind, stride, X_train_seq, X_train_dnase, y_train)