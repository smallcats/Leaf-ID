import numpy as np
import tensorflow as tf
from matplotlib import image as im
from pathlib import Path
from time import time

t0 = time()

#-----------------------------------Constants----------------------------------

d1 = 3264
d2 = 2448
learning_rate = 0.1
proj_path = Path('C:/Users/Geoffrey/Documents/GitHub/MLproject')
batch_size = 1
batches = 10

#---------------------------------Load Filenames-------------------------------

leaf_files = {'C':list(proj_path.glob('Leaf-Control/*.jpg')), 'S':list(proj_path.glob('Leaves/*.jpg'))}

#-----------------------------------Inference----------------------------------

#Inputs and labels
X = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32)

#Model parameters
K1 = tf.Variable(tf.random_normal([3,3,3,1], stddev=0.1))
b1 = tf.Variable(0.0)
W2 = tf.Variable(tf.random_normal([d1*d2], stddev=0.1))
b2 = tf.Variable(0.0)

#Convolutional layer
y1 = tf.nn.sigmoid(tf.nn.conv2d(X,K1,strides = [1,1,1,1], padding = 'SAME')+b1)

#Logistic regression layer
z = tf.reshape(y1,[d1*d2])*W2+b2
h = tf.sigmoid(z)

#--------------------------------------Cost------------------------------------

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=z)
ce_mean = tf.reduce_mean(cross_entropy)

#------------------------------------Training----------------------------------

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(ce_mean)

#-----------------------------------Import Data--------------------------------

def normalizeSamp(samplearr, new_size):
    if samplearr.shape[0] < samplearr.shape[1]: samplearr = samplearr.swapaxes(0,1)
    return np.resize(samplearr, new_size)

#-----------------------------------Run Training-------------------------------

print('time to start of training %g'%(time()-t0))
t0 = time()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

Clen = len(leaf_files['C'])
Slen = len(leaf_files['S'])
test_size = (Clen+Slen)//9

perm_idx = np.random.permutation(range(Clen+Slen))
test_idx = perm_idx[:test_size]
valid_idx = perm_idx[test_size:2*test_size]
train_idx = perm_idx[2*test_size:]

print('time to running batches %g'%(time()-t0))

for batch_num in range(batches):
    t0 = time()
    sample_idx = np.random.choice(train_idx,batch_size)
    imgs = []
    labels = []
    for k in sample_idx:
        if k < Clen:
            imgs.append(normalizeSamp(im.imread(str(leaf_files['C'][k])).astype(np.float32),(3264,2448,3)))
            labels.append(0)
        else:
            imgs.append(normalizeSamp(im.imread(str(leaf_files['S'][k-Clen])).astype(np.float32),(3264,2448,3)))
            labels.append(1)
    print('time to prep batch %d %g'%(batch_num,time()-t0))
    t0 = time()
    sess.run(train_step, {X:imgs,t:labels})
    print('time to run batch %d %g'%(batch_num,time()-t0))
