import numpy as np
import tensorflow as tf
from matplotlib import image as im
from pathlib import Path
#from time import time

#-----------------------------------Constants----------------------------------

d1 = 3264//3
d2 = 2448//3
#learning_rate = 0.1
proj_path = Path('C:/Users/Geoffrey/Documents/GitHub/MLproject')
batch_size = 1
batches = 300

#-----------------------------------Functions----------------------------------

def normalizeSamp(samplearr, new_size):
    if samplearr.shape[0] < samplearr.shape[1]: samplearr = samplearr.swapaxes(0,1)
    return samplearr.astype(np.float32)

def train_validate_test_indices(tot_samp):
    step = (tot_samp)//9
    perm_idx = np.random.permutation(range(tot_samp))
    test_idx = perm_idx[:step]
    valid_idx = perm_idx[step:2*step]
    train_idx = perm_idx[2*step:]
    return (train_idx, valid_idx, test_idx)
    
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

#Preprocessing layer
average = tf.nn.avg_pool(X,[1,3,3,1],strides=[1,3,3,1],padding='VALID')

#Convolutional layer
y1 = tf.nn.sigmoid(tf.nn.conv2d(average,K1,strides = [1,1,1,1], padding = 'SAME')+b1)

#Logistic regression layer
z = tf.reshape(y1,[d1*d2])*W2+b2
h = tf.sigmoid(z)

#--------------------------------------Cost------------------------------------

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=z)
ce_mean = tf.reduce_mean(cross_entropy)

#------------------------------------Training----------------------------------

optimizer = tf.train.AdamOptimizer(epsilon = 1e-4)
train_step = optimizer.minimize(ce_mean)

#----------------------------------Run Training--------------------------------

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

Clen = len(leaf_files['C'])
Slen = len(leaf_files['S'])

train_idx, valid_idx, test_idx = train_validate_test_indices(Clen+Slen)

for batch_num in range(batches):
    print('batch: %d'%batch_num, end='\r',flush=True)
    sample_idx = np.random.choice(train_idx,batch_size)
    imgs = []
    labels = []
    for k in sample_idx:
        if k < Clen:
#            t0 = time()
            imgs.append(normalizeSamp(im.imread(str(leaf_files['C'][k])),(d1,d2,3)))
#            print('Read+Normalization time: %f'%(time()-t0))
            labels.append(0)
        else:
#            t0 = time()
            imgs.append(normalizeSamp(im.imread(str(leaf_files['S'][k-Clen])),(d1,d2,3)))
#            print('Read+Normalization time: %f'%(time()-t0))
            labels.append(1)
#    t0 = time()
    sess.run(train_step, {X:imgs,t:labels})
#    print('Runtime: %f'%(time()-t0))

#----------------------------------Save Output---------------------------------

out = {'K1':K1.eval(sess), 'b1':b1.eval(sess), 'W2':W2.eval(sess), 'b2':b2.eval(sess)}
