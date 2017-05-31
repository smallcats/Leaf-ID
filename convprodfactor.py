import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#-------------------------------------------------Basic Functions---------------------------------------------
def padfilt(Filter, padsize0, padsize1):
    return tf.pad(Filter, [[padsize0,padsize0],[padsize1,padsize1],[0,0],[0,0]])

def conv2dmult(A,B):
    """
    Takes tensors A, B representing matrices of filters with shape [filtersize1, filtersize2, out_dim, in_dim]
      and returns the matrix product, where filters are multiplied via convolution.
    """
    Bpad = padfilt(B, tf.shape(A)[0]-1, tf.shape(A)[1]-1)
    A_whio = tf.transpose(A, (0,1,3,2))
    Bpad_nwhc = tf.transpose(Bpad, (3,0,1,2))
    C_nwhc = tf.nn.conv2d(Bpad_nwhc, A_whio, (1,1,1,1), 'VALID')
    return tf.transpose(C_nwhc, (1,2,3,0))

def padsqfilt(Filter, padsize):
    return padfilt(Filter, padsize, padsize)

def padequalize(A, B, padsize):
    return (padsqfilt(A,padsize),B)
#---------------------------------------------------Graph Inputs----------------------------------------------

def initdims(c=[1,1,1],k=[1,1,1]):
    F = tf.placeholder(tf.float32, shape=(k[0],k[0],c[0],c[2]))
    A = tf.Variable(np.random.randn(k[1],k[1],c[0],c[1]), dtype=tf.float32)
    B = tf.Variable(np.random.randn(k[2],k[2],c[1],c[2]), dtype=tf.float32)
    return (F,A,B)

#---------------------------------------------------Main Function---------------------------------------------

def filterFactorer(Farr, c=[0,0,0], k=[0,0,0], maxIter = 1000):
    """
    Factors Farr as A*B. "*" here is the product of matrices whose entries are filters.
    Tensor shape: [filter_size, filter_size, in_channels, out_channels]
    Requires: square filters, c[0], c[2] the in and out channel sizes for F, k[0] the filter size of F, and k[0]+1 = k[1]+k[2] mod 2.
    """
    #useful constants
    newFsize = k[1]+k[2]-1
    resultpadsize = (newFsize-k[0])//2
    #requirements
    if c[0] != Farr.shape[2] or c[2] != Farr.shape[3]: raise ValueError('Incompatible channel sizes')
    if (k[1]+k[2]-1-k[0])%2 != 0 or k[0] != Farr.shape[0]: raise ValueError('Incompatible filter sizes')
    #set up basic nodes
    F,A,B = initdims(c,k)
    convProd = conv2dmult(A,B)
    #pad for comparison
    if newFsize >= k[0]: Fpad, convProdpad = padequalize(F,convProd,resultpadsize)
    else: convProdpad, Fpad = padequalize(convProd,F,-resultpadsize)
    #training setup
    loss = tf.losses.mean_squared_error(Fpad,convProdpad)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(loss)
    #run training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    accuracy = []
    beststep = 0
    bestac = sess.run(loss, {F:Farr})
    best = [sess.run(A), sess.run(B)]
    for step in range(maxIter):
        newac = sess.run(loss, {F:Farr})
        if newac < bestac:
            best = [sess.run(A), sess.run(B)]
            bestac = newac
            beststep = step
        accuracy.append(newac)
        sess.run(train_step,{F:Farr})
    #outputs
    Aarr = sess.run(A)
    Barr = sess.run(B)
    bestA,bestB = best
    sess.close()
    return (Aarr,Barr, bestA, bestB, accuracy)

#-----------------------------------------------------Testing-------------------------------------------------
##F = np.random.randn(3,3,10,3)
##k = (3,3,3)
##c = (10,12,3)
##x = [k for k in range(5000)]
##
##A,B,bA,bB,acc = filterFactorer(F,c,k,5000)
##
##plt.plot(x,acc)
##plt.show()

def convprod(Aarr,Barr):
    A = tf.constant(Aarr, dtype=tf.float32)
    B = tf.constant(Barr, dtype=tf.float32)
    C = conv2dmult(A,B)
    sess = tf.Session()
    Carr = sess.run(C)
    sess.close()
    return Carr

def losses(F, A, B):
    C = convprod(A,B)
    return np.average(np.power(F-C,2)), C

def padfiltarr(Aarr, padsize):
    A = tf.constant(Aarr, dtype=tf.float32)
    Apad = padsqfilt(A,padsize)
    sess = tf.Session()
    Apadarr = sess.run(Apad)
    sess.close()
    return Apadarr
