import numpy as np
import tensorflow as tf

from os import chdir as cd
from os import listdir as ls
from os.path import isdir

from matplotlib import pyplot as plt
from PIL import Image as im

from random import shuffle

cd('C:\\Users\\Geoffrey\\Documents\\GitHub\\MLproject')

#-------------------------------------getting filenames-------------------------------------

def lspath(path):
    """
    lists the files in a directory, with path relative to current directory

    args: path: the relative path to a directory
    """
    filenames = []
    for name in ls(path):
        filenames.append(path+'\\'+name)
    return filenames

def getfilenames(path):
    filequeue = lspath(path)
    filenames = []
    while filequeue != []:
        filename = filequeue.pop()
        if isdir(filename): filequeue.extend(lspath(filename))
        elif filename[-4:] == '.png': filenames.append(filename)
    return filenames

filenames = getfilenames('.\\PreprocessedLeaves\\leafsnap-lab')
shuffle(filenames)

#-------------------------------------activation-------------------------------------

def selu(x):
    a10 = 1.6733
    l10 = 1.0507
    halfgap = (l10 - a10*l10)/2
    center = (l10 + a10*l10)/2
    return tf.nn.elu(x)*(halfgap*tf.sign(x)+center)

#-------------------------------------nn graph builders-------------------------------------

def addFullLayer(inlayer, outdim, nonlinearity, initialization=np.random.standard_normal, identifier='', final = False):
    """
    Adds a fully connected layer. Inlayer is a tensor of shape [batchsize, layersize]
    """
    Winit = (1/int(inlayer.shape[1]))*initialization((int(inlayer.shape[1]),outdim))
    binit = 0.1*initialization(outdim)
    W = tf.Variable(Winit, dtype=tf.float32, name='W'+identifier)
    b = tf.Variable(binit, dtype=tf.float32, name='b'+identifier)
    if final == False: L = nonlinearity(tf.matmul(inlayer,W)+b)
    else: L = 255*tf.sigmoid(1.6*tf.matmul(inlayer,W)+b)
    return W,b,L

def buildCore(inlayer, layers, nonlinearity, initialization=np.random.standard_normal):
    """
    Builds TensorFlow operations for the layers of an autoencoder. No input, training, saving, etc.
    args: inlayer - a 2D tensor which will serve as the input layer. Typically the shape is [batch, features].
    """
    W,b,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    Ws = [W]
    bs = [b]
    Ls = [L]
    k = 1
    for l in layers[1:]:
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        Ls.append(L)
        k += 1
    for l in layers[-2::-1]:
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        Ls.append(L)
        k += 1
    W,b,L = addFullLayer(Ls[-1],inlayer.shape[1], nonlinearity, initialization, str(k), final=True)
    Ws.append(W)
    bs.append(b)
    Ls.append(L)
    return Ws, bs, Ls

def buildCoreEnc(inlayer, layers, nonlinearity, initialization=np.random.standard_normal):
    """
    Builds a Tensorflow graph for the layers of the encoding portion of an autoencoder.

    args: inlayer: a tensor input layer
          layers: a list of sizes of encoding layers
          nonlinearity: the activation function
          initialization: a function for initialization, should take a tuple for its size argument
    """
    W,b,L = addFullLayer(inlayer, layers[0], nonlinearity, initialization, '0')
    Ws = [W]
    bs = [b]
    Ls = [L]
    k = 1
    for l in layers[1:]:
        W,b,L = addFullLayer(Ls[-1],l,nonlinearity,initialization,str(k))
        Ws.append(W)
        bs.append(b)
        Ls.append(L)
        k += 1
    return Ws, bs, Ls

#-------------------------------------training-------------------------------------

def buildTraining(t_filenames, v_filenames, layers=[1000,100], nonlinearity=tf.nn.sigmoid, optimize=tf.train.GradientDescentOptimizer(0.1), lossfunc=tf.losses.mean_squared_error):
    """
    Builds a TensorFlow autoencoder graph for 90x60 images including training, saving, input, initialization, etc.

    args: t_filenames, v_filenames, layers, nonlinearity, lossfunc
    
    returns: choice: a placeholder node for choosing whether to run on the training or validation set.
             trainstep: the training step node
             loss: the loss node
             init: the initializer
             saver: the Saver node
             coord: the coordinator
             numfiles: the number of files in the training set
    """
    reader = tf.WholeFileReader()
    
    #getting training leaves
    t_imagequeue = tf.train.string_input_producer(t_filenames)
    t_label, t_imgstr = reader.read(t_imagequeue)
    t_imgtensor = tf.image.decode_png(t_imgstr)
    t_reshape_img = tf.cast(tf.reshape(t_imgtensor, [60*90*3]), tf.float32)
    t_img_batch = tf.train.shuffle_batch([t_reshape_img], batch_size = 20, capacity = 100, min_after_dequeue = 40)

    #getting validation leaves
    v_imagequeue = tf.train.string_input_producer(v_filenames)
    v_label, v_imgstr = reader.read(v_imagequeue)
    v_imgtensor = tf.image.decode_png(v_imgstr)
    v_reshape_img = tf.cast(tf.reshape(v_imgtensor, [60*90*3]), tf.float32)
    v_img_batch = tf.train.shuffle_batch([v_reshape_img], batch_size = 50, capacity = 250, min_after_dequeue = 100)

    #splitter - decide to run on training or validation set
    choice = tf.placeholder(dtype=tf.bool) #True for training, False for validation
    in_imgs = tf.cond(choice, lambda: t_img_batch, lambda: v_img_batch)

    #build layers
    Lin = in_imgs/255
    Ws, bs, Ls = buildCore(Lin, layers, nonlinearity)

    #training
    loss = lossfunc(in_imgs, Ls[-1])
    optimizer = optimize
    trainstep = optimizer.minimize(loss)

    #initialization and saving
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    return choice, trainstep, loss, init, saver, coord

def runTraining(choice, trainstep, loss, init, saver, coord, name = 'model', numsteps=100, validation_steps=100):
    """
    Trains an autoencoder graph for numsteps steps, saving the resulting weights in a .ckpt file.
    Requires: imagename, trainstep, loss, coord, init, saver all with those names.
    
    args: name: a string for saving the model
          numsteps: the number of steps of training
          validation_steps: the number of training steps before validation
          nodes from buildTraining graph - choice, trainstep, loss, init, saver, coord
          
    returns: t_losses: a list of losses for each training step
             v_losses: a list of losses for each validation
    """
    sess = tf.Session()
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    t_losses = []
    v_losses = []
    for step in range(numsteps):
        _, lossstep = sess.run((trainstep,loss), {choice: True})
        t_losses.append(lossstep)
        if (step+1)%(numsteps//validation_steps) == 0:
            lossstep = sess.run(loss, {choice: False})
            v_losses.append(lossstep)
            print('.',sep='',end='')
    print('\n')
    
    saver.save(sess, '.\\Models\\'+name+'.ckpt')

    coord.request_stop()
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()
    return t_losses, v_losses

#-------------------------------------encode-decode-------------------------------------

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
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)/255
    coord = tf.train.Coordinator()

    #build layers
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
    Lin = tf.cast(tf.reshape(imgtensor, [-1, 60*90*3]), tf.float32)/300
    coord = tf.train.Coordinator()

    #build layers
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

#-------------------------------------main-------------------------------------

##inimg = im.fromarray(image, 'RGB')
##outimg = im.fromarray(np.uint8(np.rint(recons_image)), 'RGB')
##
##x = np.arange(1,numsteps+1)
##plt.plot(x, losses)
##plt.xlabel('iteration')
##plt.ylabel('loss')
##plt.show()
