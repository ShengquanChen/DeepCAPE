#!/usr/bin/env python
# encoding: utf-8
import os
gpu_use = raw_input('gpu:')
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import numpy as np  
import tensorflow as tf
#tf_config = tf.ConfigProto()
#tf_config.gpu_options.allow_growth=True
#tf.Session(config=tf_config)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization  
import hickle as hkl 

def ae_train(name, stride, ratio, ind, expr_num, encoding_dim):
    # Load DNase-seq data of training set.
    dataset = './data1_%d/%s_train_stride%d_%d_dnase.hkl'%(ratio, name, stride, ind)
    X_train_dnase, y_train = hkl.load(dataset)
    X_train_dnase = X_train_dnase.reshape(-1, expr_num*300)
    print('[X_train_dnase shape: {}, y_train shape: {}]'.format(X_train_dnase.shape, y_train.shape))
    # Load DNase-seq data of test set.
    dataset = './data1_%d/%s_test_stride%d_%d_dnase.hkl'%(ratio, name, stride, ind)
    X_test_dnase, y_test = hkl.load(dataset)
    X_test_dnase = X_test_dnase.reshape(-1, expr_num*300)
    print('[X_test_dnase shape: {}, y_test shape: {}]'.format(X_test_dnase.shape, y_test.shape))
    
    # Auto-encoder model.
    input_img = Input(shape=(expr_num*300,))  
    encoded_output = BatchNormalization(epsilon=0.0001, mode=1, momentum=0.9)(input_img)
	encoded_output = Dense(300, activation='relu')(encoded_output)
    decoded_input  = BatchNormalization(epsilon=0.0001, mode=1, momentum=0.9)(encoded_output)
	decoded_output = Dense(expr_num*300, activation='tanh')(decoded_input)
	autoencoder = Model(input=input_img, output=decoded_output)
	encoder = Model(input=input_img, output=encoded_output)
    
    # Train the auto-encoder model.
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    autoencoder.compile(optimizer=adam, loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', verbose=0, patience=2, mode='min')
    filename = './AE_weight/ratio%d_%s_stride%d_%d.h5' % (ratio, name, stride, ind)
    save_best = ModelCheckpoint(filename, save_best_only=True, save_weights_only=True)
    if os.path.isfile(filename):
    	print '[Loading]' + filename
    	autoencoder.load_weights(filename)
    x_train = X_train_dnase
    autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=128, shuffle=True, validation_split=0.1, callbacks=[early_stopping, save_best])

    # Encode the DNase-seq data.
    dnase_train_X = encoder.predict(X_train_dnase).reshape(-1, 1, 300)
    dnase_test_X  = encoder.predict(X_test_dnase).reshape(-1, 1, 300)
    hkl.dump((dnase_train_X, y_train), './data1_%d/%s_train_stride%d_%d_dnase_AE.hkl' % (ratio, name, stride, ind), 'w')
    hkl.dump((dnase_test_X, y_test), './data1_%d/%s_test_stride%d_%d_dnase_AE.hkl' % (ratio, name, stride, ind), 'w')

if __name__ == "__main__":
    names = ['epithelial_cell_of_esophagus','melanocyte','cardiac_fibroblast','keratinocyte','myoblast','stromal','mesenchymal','natural_killer','monocyte']
    print names
    stride = input('Choose stride : ')
    encoding_dim = 300
    for name in names:
        if name == 'mesenchymal':
            expr_num = 2
        if name == 'monocyte':
            expr_num = 3
        if name == 'keratinocyte':
            expr_num = 4
        if name == 'myoblast':
            expr_num = 2
        if name == 'melanocyte':
            expr_num = 1
        if name == 'natural_killer':
            expr_num = 1
        if name == 'stromal':
            expr_num = 4
        if name == 'epithelial_cell_of_esophagus':
            expr_num = 2
        if name == 'cardiac_fibroblast':
            expr_num = 2
        for ind in range(5):
            for ratio in [10, 20]:
                ae_train(name, stride, ratio, ind, expr_num, encoding_dim)