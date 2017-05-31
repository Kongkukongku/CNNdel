#coding:utf-8  
""" 
Author:salarmacata 
Source:https://github.com/JingWCrystal/CNNdel
 
"""  
from __future__ import absolute_import  
from __future__ import print_function  
from keras.preprocessing.sequence import *  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils  
from six.moves import range  
# from data import load_data  
# from data import load_test
import random  
import theano
import numpy as np
import getopt
import os 
import sys 
import getopt

neurons_stats =0
shuffle_mode =0
activation_relu=0
learning_rate=0.1
neurons=256
batch = 32
epoch = 50
def shuffle_input(X_train,train_label,train_cnt):
	index = [i for i in range(train_cnt)]  
	random.shuffle(index)  
	X_train = X_train[index]  
	train_label = train_label[index]

def load_data(train_data,train_tag):  
    train_arr = np.loadtxt(train_data,dtype=np.float64)  
    train_label = np.loadtxt(train_tag,dtype=np.int8) 
    train_samples = train_arr.shape[0]
    train_feature = train_arr.shape[1]
    newshape_train = (train_samples,1,1,train_feature)
    X_train = np.reshape(train_arr,newshape_train)
    print(train_arr[0])
    print(train_label[0])
    return X_train,train_label,train_samples

def load_test(test_path):	  
	test_arr = np.loadtxt(test_path,dtype=np.float64)
	test_samples = test_arr.shape[0]
	test_feature = test_arr.shape[1]
	newshape_test = (test_samples,1,1,test_feature)
	X_test = np.reshape(test_arr,newshape_test)
	return X_test
def usage():
	if len(sys.argv) <= 4:
		print('Usage: [options] train_data train_tag test_path ')
		print('\tsample: python3 CNNdel.py -o /mnt/hdd/DL/res CNNtrain.data CNN_train_lable.txt /mnt/hdd/DL/test ')
		print('options:')
		print('-l learning rate: set learning rate of the CNNs')
		print('\t -- default 0.1 ')
		print('-b batch: set batch of train input')
		print('\t -- default 32 ')
		print('-e epoch: set epoch of train input')
		print('\t -- default 50 ')
		print('-a activation: set relu as activation of CNNs')
		print('\t -- default tanh ')
		print('-s shuffle: set shuffle mode open for train process')
		print('\t -- default no shuffle ')
		print('-n neurons : set number of neurons in flatten layer')
		print('\t -- default 256. ')
		print('-o output file directory')
		raise SystemExit

try:
	opts, args = getopt.getopt(sys.argv[1:], "hl:b:e:aso:")
	print(opts)
	print(args)

except getopt.GetoptError:
	sys.exit()
# if len(opts) == 0:
# 	print('opts=0!')
# 	usage()
# 	sys.exit()
for opt, arg in opts:
	if opt == '-l':
		learning_rate=float(arg)
	elif opt == '-b':
		batch=int(arg)
	elif opt == '-e':
		epoch=int(arg)
	elif opt == '-a':
		activation_relu =1
	elif opt == '-o':
		out_path=arg
	elif opt == '-s':
		shuffle_mode =1
	elif opt == '-n':
		neurons_stats =1
		neurons = int(arg)
	elif opt == '-h':
		usage()
		sys.exit()

train_data=args[0]
train_tag=args[1]
test_path=args[2]
print(learning_rate,batch,epoch,activation_relu,out_path,shuffle_mode,neurons_stats,neurons)
print(train_data)
print(train_tag)
print(test_path)


X_train,train_label,train_cnt= load_data(train_data,train_tag)
print('X_train shape:', X_train.shape)
print('label shape:', train_label.shape)


# if shuffle_mode == 1:
# 	shuffle_input(X_train,train_label,train_cnt)

train_label = np_utils.to_categorical(train_label, 2)  
print('Build model...')
model = Sequential()  
model.add(Convolution2D(4,1,4,input_shape=(1,1,49),init='uniform', weights=None,border_mode='valid'))

model.add(Activation('tanh'))
# if activation_relu == 1:
# 	model.add(Activation('relu')) 
# else:
# 	model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Convolution2D(8,1,4, border_mode='valid')) 
model.add(Activation('tanh')) 
# if activation_relu == 1:
# 	model.add(Activation('relu')) 
# else:
# 	model.add(Activation('tanh')) 
model.add(MaxPooling2D(pool_size=(1, 2)))
 
model.add(Flatten())  
model.add(Dense(256,init='normal'))
model.add(Activation('tanh'))
# if neurons_stats == 1:
# 	model.add(Dense(neurons,init='normal'))
# else:
# 	model.add(Dense(256,init='normal'))
# if activation_relu == 1:
# 	model.add(Activation('relu')) 
# else:
# 	model.add(Activation('tanh'))

model.add(Dense(2, init='normal'))  
model.add(Activation('softmax'))  
model.summary()

#sgd = SGD(lr=learning_rate, decay=0.00001, momentum=0.1, nesterov=False)  
sgd = SGD(lr=0.1, decay=0.00001, momentum=0.1, nesterov=False)  
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=sgd)  
model.fit(X_train,train_label, batch_size=32, nb_epoch=10,shuffle=False,verbose=2,validation_split=0.2)

# if shuffle_mode ==0:
# 	
# 	sys.exit()
# else:
# 	model.fit(X_train,train_label, batch_size=batch, nb_epoch=epoch,shuffle=True,verbose=1,validation_split=0.2)

# for test_sample in open("/mnt/hdd/DL/list/sample_chr_test.list"):  
# 	test_sample = test_sample.strip('\n')
test_list=os.listdir(test_path)
print(test_list)
for sample_chr_test_data in test_list:    #get the test file name like 'NA19984.chrom20.test.data'
	print(sample_chr_test_data)
	sample_chr_test_data_path=test_path+'/'+sample_chr_test_data
	print(sample_chr_test_data_path)
	X_test =load_test(sample_chr_test_data_path)

	print('X_test shape:', X_test.shape)
	sample_chr=sample_chr_test_data[0:15] # get the substring like 'NA19984.chrom20'
	print(sample_chr)
	result = model.predict(X_test)
	res2 = np.around(result)
	out_file=out_path+"/"+sample_chr+".predict.res"
	print(out_file)
	np.savetxt(out_file,res2[:,1],fmt='%d')
	file_commond='sh cnn_file.sh'+" "+sample_chr+" "+out_path
	print(file_commond)
	os.system(file_commond)









