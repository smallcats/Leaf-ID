import numpy as np
import tensorflow as tf
from os import chdir as cd
from os import listdir as ls
from matplotlib import pyplot as plt
from PIL import Image as im

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')

def autoenc(layers=[1000,100], numsteps=100, nonlinearity=tf.nn.sigmoid, optimize=tf.train.GradientDescentOptimizer(0.1), lossfunc=tf.losses.mean_squared_error):
    #getting leaves
    def getfilenames(path):
        filenames = []
        for name in ls(path):
            filenames.append(path+'\\'+name)
        return filenames

    imagequeue = tf.train.string_input_producer(getfilenames('.\\PreprocessedLeaves'))
    reader = tf.WholeFileReader()
    _, imgstr = reader.read(imagequeue)
    imgtensor = tf.image.decode_png(imgstr)
    gin = tf.Variable(np.random.rand(), dtype=tf.float32)
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)*(gin/300)

    #build layers
    def addFullLayer(inlayer, outdim, identifier='', final = False):
        """
        Adds a fully connected layer. Inlayer is a tensor of shape [batchsize, layersize]
        """
        W = tf.Variable((1/int(outdim))*np.random.randn(inlayer.shape[1],outdim), dtype=tf.float32, name='W'+identifier)
        b = tf.Variable(0.1*np.random.randn(outdim), dtype=tf.float32, name='b'+identifier)
        g = tf.Variable(np.random.rand(), dtype=tf.float32)
        if final == False: L = nonlinearity(g*(tf.matmul(inlayer,W)+b))
        else: L = tf.matmul(inlayer,W)+b
        return W,b,g,L

    def buildautoenc(inlayer, layersizes):
        W,b,g,L = addFullLayer(inlayer, layersizes[0], '1')
        Ws = [W]
        bs = [b]
        gs = [g]
        Ls = [L]
        k = 1
        for l in layersizes[1:]:
            W, b, g, L = addFullLayer(Ls[-1], l, str(k))
            Ws.append(W)
            bs.append(b)
            gs.append(g)
            Ls.append(L)
            k += 1
        for l in layersizes[-2::-1]:
            W,b,g,L = addFullLayer(Ls[-1],l,str(k))
            Ws.append(W)
            bs.append(b)
            gs.append(g)
            Ls.append(L)
            k += 1
        W,b,g,L = addFullLayer(Ls[-1],inlayer.shape[1], str(k), final=True)
        Ws.append(W)
        bs.append(b)
        gs.append(g)
        Ls.append(L)
        return Ws, bs, gs, Ls

    Ws, bs, gs, Ls = buildautoenc(Lin, [1000,100])
    ##W1, b1, L1 = addFullLayer(Lin, 1000,'1')
    ##W2, b2, L2 = addFullLayer(L1, 100,'2')
    ##W3, b3, L3 = addFullLayer(L2, 1000,'3')
    ##W4, b4, Lout = addFullLayer(L3, 60*90*3,'4',True)
    outimg = tf.reshape(Ls[-1], (90,60,3))

    #training
    saver = tf.train.Saver()
    loss = tf.losses.mean_squared_error(outimg, imgtensor)
    optimizer = optimize
    train_step = optimizer.minimize(loss)

    #initialize and run
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    sess = tf.Session()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(init)
    losses = []
    for step in range(numsteps): #2 for testing, should be len(ls(.\\Leaves))
        _, lossstep = sess.run((train_step,loss))
        losses.append(lossstep)
        if step%(numsteps//100) == 0: print('.',sep='',end='')

    saver.save(sess, '.\\Models\\testbigmodel.ckpt')
    image, recons_image, lossnow = sess.run((imgtensor, outimg, loss))

    inimg = im.fromarray(image, 'RGB')
    outimg = im.fromarray(np.uint8(np.rint(recons_image)), 'RGB')

    coord.request_stop()
    coord.join(threads)
    sess.close()

    x = np.arange(1,numsteps+1)
    plt.plot(x, losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    return losses, lossnow, inimg, outimg
