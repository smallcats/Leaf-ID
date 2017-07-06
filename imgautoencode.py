import numpy as np
import tensorflow as tf
from os import chdir as cd
from os import listdir as ls
from matplotlib import pyplot as plt
from PIL import Image as im

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')

def getfilenames(path):
    filenames = []
    for name in ls(path):
        filenames.append(path+'\\'+name)
    return filenames

filenames = getfilenames('.\\PreprocessedLeaves')

def selu(x):
    a10 = 1.6733
    l10 = 1.0507
    halfgap = (l10 - a10*l10)/2
    center = (l10 + a10*l10)/2
    return tf.nn.elu(x)*(halfgap*tf.sign(x)+center)

def addFullLayer(inlayer, outdim, nonlinearity, initialization=np.random.standard_normal, identifier='', final = False):
    """
    Adds a fully connected layer. Inlayer is a tensor of shape [batchsize, layersize]
    """
    Winit = (1/int(inlayer.shape[1]))*initialization((int(inlayer.shape[1]),outdim))
    binit = 0.1*initialization(outdim)
    #ginit = np.random.uniform()
    W = tf.Variable(Winit, dtype=tf.float32, name='W'+identifier)
    b = tf.Variable(binit, dtype=tf.float32, name='b'+identifier)
    #g = tf.Variable(ginit, dtype=tf.float32)
    #if final == False: L = nonlinearity(g*(tf.matmul(inlayer,W)+b))
    if final == False: L = nonlinearity(tf.matmul(inlayer,W)+b)
    else: L = 255*tf.sigmoid(tf.matmul(inlayer,W)+b)
    #return W,b,g,L
    return W,b,L

def buildCore(inlayer, layers, nonlinearity, initialization=np.random.standard_normal):
    """
    Builds TensorFlow operations for the layers of an autoencoder. No input, training, saving, etc.
    args: inlayer - a 2D tensor which will serve as the input layer. Typically the shape is [batch, features].
    """
    W,b,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    #W,b,g,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    Ws = [W]
    bs = [b]
    #gs = [g]
    Ls = [L]
    k = 1
    for l in layers[1:]:
        #W,b,g,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        #gs.append(g)
        Ls.append(L)
        k += 1
    for l in layers[-2::-1]:
        #W,b,g,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        #gs.append(g)
        Ls.append(L)
        k += 1
    #W,b,g,L = addFullLayer(Ls[-1],inlayer.shape[1], nonlinearity, initialization, str(k), final=True)
    W,b,L = addFullLayer(Ls[-1],inlayer.shape[1], nonlinearity, initialization, str(k), final=True)
    Ws.append(W)
    bs.append(b)
    #gs.append(g)
    Ls.append(L)
    #return Ws, bs, gs, Ls
    return Ws, bs, Ls

def buildCoreEnc(inlayer, layers, nonlinearity, initialization=np.random.standard_normal):
    """
    Builds a Tensorflow graph for the layers of the encoding portion of an autoencoder.

    args: inlayer: a tensor input layer
          layers: a list of sizes of encoding layers
          nonlinearity: the activation function
          initialization: a function for initialization, should take a tuple for its size argument
    """
    #W,b,g,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    W,b,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    Ws = [W]
    bs = [b]
    #gs = [g]
    Ls = [L]
    k = 1
    for l in layers[1:]:
        #W,b,g,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        #gs.append(g)
        Ls.append(L)
        k += 1
    #return Ws, bs, gs, Ls
    return Ws, bs, Ls

def buildTraining(filenames, layers=[1000,100], nonlinearity=tf.nn.sigmoid, optimize=tf.train.GradientDescentOptimizer(0.1), lossfunc=tf.losses.mean_squared_error):
    """
    Builds a TensorFlow autoencoder graph for 90x60 images including training, saving, input, initialization, etc.
    Returns: trainstep: the training step node
             loss: the loss node
             init: the initializer
             saver: the Saver node
             coord: the coordinator
             numfiles: the number of files in the training set
    """
    #getting leaves
    imagequeue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.WholeFileReader()
    _, imgstr = reader.read(imagequeue)
    imgtensor = tf.image.decode_png(imgstr)
    #gin = tf.Variable(np.random.rand(), dtype=tf.float32, name='gin')
    #Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)*(gin/300)
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)/300

    #build layers
    #Ws, bs, gs, Ls = buildCore(Lin, layers, nonlinearity)
    Ws, bs, Ls = buildCore(Lin, layers, nonlinearity)
    outimg = tf.reshape(Ls[-1], (90,60,3))

    #training
    loss = lossfunc(outimg, imgtensor)
    optimizer = optimize
    trainstep = optimizer.minimize(loss)

    #initialization and saving
    #saver = tf.train.Saver({k.name:k for k in Ws+bs+gs}.update({gin.name:gin}))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    return trainstep, loss, init, saver, coord

def runTraining(trainstep, loss, init, saver, coord, name = 'model', numsteps=100):
    """
    Trains an autoencoder graph for numsteps steps, saving the resulting weights in a .ckpt file.
    Requires: imagename, trainstep, loss, coord, init, saver all with those names.
    
    args: name: a string for saving the model
          numsteps: the number of steps of training
          trainstep
          loss
          init
          saver
          coord
          
    returns: losses: a list of losses for each training step
    """
    sess = tf.Session()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(init)
    losses = []
    if numsteps >= 100:
        for step in range(numsteps):
            _, lossstep = sess.run((trainstep,loss))
            losses.append(lossstep)
            if step%(numsteps//100) == 0: print('.',sep='',end='')
    else:
        for step in range(numsteps):
            _, lossstep = sess.run((trainstep,loss))
            losses.append(lossstep)
            print('.',sep='',end='')
    print('\n')
    
    saver.save(sess, '.\\Models\\'+name+'.ckpt')

    coord.request_stop()
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()
    return losses

def buildEncDecInference(filenames, layers=[1000,100], nonlinearity=tf.nn.sigmoid, lossfunc=tf.losses.mean_squared_error):
    """
    Builds an inference graph for the operation sequence encode,decode for an autoencoder trained on 90x60 pixel images.

    args: filenames: a list of file names on which to run inference
          layers: list of the sizes of encoding layers after the input layer
          nonlinearity: the activation function
          lossfunc: the loss function

    returns: outimg: the output tensor (reshaped to [90,60,3], dtype=tf.float32)
             loss: the loss tensor
             init: the initializer
             restorer: a Saver object for restoring weights
             coord: the Coordinator
             numfiles: then number of image files to be run
    """
    #getting leaves
    numfiles = len(filenames)
    imagequeue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.WholeFileReader()
    _, imgstr = reader.read(imagequeue)
    imgtensor = tf.image.decode_png(imgstr)
    #gin = tf.Variable(0, dtype=tf.float32, name='gin')
    #Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)*(gin/300)
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)/300
    coord = tf.train.Coordinator()

    #build layers
    #Ws, bs, gs, Ls = buildCore(Lin, layers, nonlinearity, initialization = np.zeros)
    Ws, bs, Ls = buildCore(Lin, layers, nonlinearity, initialization = np.zeros)
    outimg = tf.reshape(Ls[-1], (90,60,3))

    #add loss
    loss = lossfunc(outimg, imgtensor)

    #for initialization
    init = tf.global_variables_initializer()
    restorer = tf.train.Saver()
    
    return outimg, loss, init, restorer, coord, numfiles

def runEncDecInference(outimg, loss, init, restorer, coord, numfiles, name='model'):
    """
    Runs an encode-decode inference graph for an autoencoder trained on 90x60 pixel images.

    args: outimg: the output image tensor
          loss: the loss tensor
          init: the initializer
          restorer: the Saver object to restore weights
          coord: the coordinator
          name: name of the trained model from which to restore variables

    returns: npoutimgs: a list of (90,60,3)-shaped uint8 numpy arrays representing the output images
             losses: a list of losses for input images
    """
    sess = tf.Session()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(init)
    restorer.restore(sess, '.\\Models\\'+name+'.ckpt')

    losses = []
    npoutimgs = []
    for step in range(numfiles):
        npoutimg, lossstep = sess.run((outimg,loss))
        losses.append(lossstep)
        npoutimgs.append(npoutimg)
        print('.',sep='',end='')

    tf.reset_default_graph()
    return npoutimgs, losses
    
def buildEncInference(filenames, layers=[1000,100], nonlinearity=tf.nn.sigmoid):
    """
    Builds a TensorFlow graph for encode inference for an autoencoder trained on 90x60 pixel images.

    args: filenames: the names of files on which inference will be run
          layers: the encode layers of the autoencoder (after the input layer)
          nonlinearity: the activation function

    returns: outlayer: the output layer tensor
             init: initializer node
             restorer: Saver node for restoring model
             coord: coordinator node
             numfiles: the number of files to be encoded
    """
    #getting leaves
    numfiles = len(filenames)
    imagequeue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.WholeFileReader()
    _, imgstr = reader.read(imagequeue)
    imgtensor = tf.image.decode_png(imgstr)
    #gin = tf.Variable(0, dtype=tf.float32, name='gin')
    #Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)*(gin/300)
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)/300
    coord = tf.train.Coordinator()

    #build layers
    #Ws, bs, gs, Ls = buildCoreEnc(Lin, layers, nonlinearity, initialization = np.zeros)
    Ws, bs, Ls = buildCoreEnc(Lin, layers, nonlinearity, initialization = np.zeros)

    #for initialization
    init = tf.global_variables_initializer()
    restorer = tf.train.Saver()
    
    return Ls[-1], init, restorer, coord, numfiles

def runEncInference(outlayer, init, restorer, coord, numfiles, name='model'):
    """
    Runs an encoding graph from an autoencoder trained on 90x60 pixel images.

    args: outlayer: the output layer
          init: initialization node
          restorer: Saver node for restoring
          coord: coordinator
          numfiles: number of files loaded into the queue in the graph
          name: name under which the model is saved

    returns: encimgs: a list containing the encoded images (as a numpy float32 array)
    """
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    restorer.restore(sess, '.\\Models\\'+name+'.ckpt')

    encimgs = []
    for step in range(numfiles):
        encimg = sess.run(outlayer)
        encimgs.append(encimg)

    sess.close()
    tf.reset_default_graph()
    return encimgs

##inimg = im.fromarray(image, 'RGB')
##outimg = im.fromarray(np.uint8(np.rint(recons_image)), 'RGB')
##
##x = np.arange(1,numsteps+1)
##plt.plot(x, losses)
##plt.xlabel('iteration')
##plt.ylabel('loss')
##plt.show()
