import numpy as np
import tensorflow as tf
from os import chdir as cd
from os import listdir as ls
from matplotlib import image as im

#getting leaves
def getfilenames(path):
    filenames = []
    for name in ls(path):
        filenames.append(path+'\\'+name)
    return filenames

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')
imagequeue = tf.train.string_input_producer(getfilenames('.\\PreprocessedLeaves'))
reader = tf.WholeFileReader()
_, imgstr = reader.read(imagequeue)
imgtensor = tf.image.decode_png(imgstr)
Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)

#build layers
def addFullLayer(inlayer, outdim):
    """
    Adds a fully connected layer. Inlayer is a tensor of shape [batchsize, layersize]
    """
    W = tf.Variable(np.random.randn(inlayer.shape[1],outdim), dtype=tf.float32)
    b = tf.Variable(np.random.randn(outdim), dtype=tf.float32)
    L = tf.nn.relu(tf.matmul(inlayer,W)+b)
    return W,b,L

W1, b1, L1 = addFullLayer(Lin, 1000)
W2, b2, L2 = addFullLayer(L1, 100)
W3, b3, L3 = addFullLayer(L2, 1000)
W4, b4, Lout = addFullLayer(L3, 60*90*3)
outimg = tf.reshape(Lout, (90,60,3))

#training
loss = tf.losses.absolute_difference(outimg, imgtensor)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

#initialize and run
init = tf.global_variables_initializer()
coord = tf.train.Coordinator()

sess = tf.Session()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
sess.run(init)
for _ in range(50): #2 for testing, should be len(ls(.\\Leaves))
    sess.run(train_step)
    print(sess.run(loss))

coord.request_stop()
coord.join(threads)
sess.close()
