import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import operator
from functools import reduce

#data preparation
data_dir = r'./cifar10_data/cifar-10-batches-bin'
data_height = 24

###A########################################AutoEncoder Part Start######################################################
##Parameter definition(AutoEncoder)##
#autoEncoder CNN parameter
learning_rate = 0.01
n_conv1 = 24
n_conv2 = 48
batch_size = 200

#input train,test data from file
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

#data placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, data_height, data_height, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

#Weights, Biases definition
weights = {
    "encoder_conv1": tf.Variable(tf.truncated_normal([3, 3, 3, n_conv1],stddev=0.1)),
    "encoder_conv2": tf.Variable(tf.random_normal([3, 3, n_conv1, n_conv2], stddev=0.1)),

    "decoder_conv1": tf.Variable(tf.random_normal([3, 3, 3, n_conv1], stddev=0.1)),
    "decoder_conv2": tf.Variable(tf.random_normal([3, 3, n_conv1, n_conv2], stddev=0.1)),
}
biases = {
    "encoder_conv1":tf.Variable(tf.zeros([n_conv1])),
    "encoder_conv2":tf.Variable(tf.zeros([n_conv2])),

    "decoder_conv1":tf.Variable(tf.zeros([n_conv1])),
    "decoder_conv2":tf.Variable(tf.zeros([n_conv2])),
}

##Function definition(AutoEncoder)##
#convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#encoding
def encoder(x):
    h_conv1 = tf.nn.relu(conv2d(x,weights["encoder_conv1"])+biases["encoder_conv1"])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights["encoder_conv2"]) + biases["encoder_conv2"])
    return h_conv2,h_conv1
#max pooling
def max_pool_with_argmax(net,stride):
    _,mask = tf.nn.max_pool_with_argmax(net,ksize = [1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")
    return net, mask
#unpooling
def unpool(net,mask,stride):
    ksize = [1,stride,stride,1]
    input_shape = net.get_shape().as_list()
    output_shape = (input_shape[0],input_shape[1]*ksize[1],input_shape[2]*ksize[2],input_shape[3])
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0],dtype=tf.int64),shape=[input_shape[0],1,1,1])
    b = one_like_mask * batch_range
    y = mask//(output_shape[2]*output_shape[3])
    x = mask % (output_shape[2]*output_shape[3])//output_shape[3]
    feature_range = tf.range(output_shape[3],dtype=tf.int64)
    f = one_like_mask * feature_range

    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net,[updates_size])
    ret = tf.scatter_nd(indices,values,output_shape)
    return ret
#decoding
def decoder(x,conv1):
    t_conv1 = tf.nn.conv2d_transpose(x-biases["decoder_conv2"],weights["decoder_conv2"],conv1.shape,[1,1,1,1])
    t_x_image = tf.nn.conv2d_transpose(t_conv1-biases["decoder_conv1"],weights["decoder_conv1"],image_holder.shape,[1,1,1,1])
    return  t_x_image

##Function Running Part(AutoEncoder)##
#Convolution(Output:batchsize*data_height*data_height*nconv_2)(Output:128*32*32*32)
encoder_out, conv1 = encoder(image_holder)
#Pooling(Output:128*16*16*32)
h_pool2, mask = max_pool_with_argmax(encoder_out,2)
#Unpooling
h_upool = unpool(h_pool2, mask, 2)
#Doconvolution
pred = decoder(h_upool,conv1)

#cost calculation
cost = tf.reduce_mean(tf.pow(image_holder - pred,2))
#Optimizer(RMSPropOptimizer)
optimzer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#Initialize session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

#Training for Autoencoder
train_step = 3000
display_step = 100
for step in range(train_step):
    start_time = time.time()
    #input train data to batch
    image_batch = sess.run(images_train)
    #training
    _, c = sess.run([optimzer, cost], feed_dict={image_holder: image_batch})
    #log duration time
    duration = time.time() - start_time
    #show step, cost, time
    if step % display_step == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step, c, examples_per_sec, sec_per_batch))

#Testing for Autoencoder(File out while classificating)
display_num = 10
#input test data to batch
image_batch = sess.run(images_test)
#get result from test data
reconstruction = sess.run(pred, feed_dict={image_holder: image_batch})
#plot test data and test result
f, a = plt.subplots(2, display_num, figsize=(display_num, 2))
for i in range(display_num):
    a[0][i].imshow(image_batch[i])
    a[1][i].imshow(reconstruction[i])
plt.draw()
plt.show()
########################################################################################################################

#############################################Classification Part(CNN)###################################################
##Parameter definition Part(Classification Part)##
#CNN parameter
learning_rate = 0.001
batch_size = batch_size # use same batchsize with Autoencoder part
cNNinput = n_conv2

keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)

#input train data, test data from file
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

#CNN image, label holder definition
image_holder2 = tf.placeholder(tf.float32, [batch_size, data_height, data_height, cNNinput])  #use data before pooling
label_holder2 = tf.placeholder(tf.float32, [batch_size])

##Function definition Part(Classification Part)##
#L2 Regularization
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#CNN layer
def add_CNN_layer(input_data, input_shape, output_shape, conv_size, stddev, weightloss):
    weight = variable_with_weight_loss(shape=[conv_size, conv_size, input_shape, output_shape],
                                       stddev=stddev, wl = weightloss)
    kernel = tf.nn.conv2d(input_data, weight, [1,1,1,1], padding='SAME')
    bias = tf.Variable(tf.constant(0.0,shape=[output_shape]))
    conv = tf.nn.relu(kernel + bias)
    pool = tf.nn.max_pool(conv, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')
    norm = tf.nn.lrn(pool,4,bias=1.0, alpha=0.001/9.0,beta=0.75)
    return norm

#fc layer
def add_fc_layer(input_data,input_shape, output_shape, const, stddev, weightloss):
    weight = variable_with_weight_loss([input_shape,output_shape],stddev=stddev,wl = weightloss)
    bias = tf.Variable(tf.constant(const,shape=[output_shape]))
    local = tf.nn.relu(tf.matmul(input_data,weight)+bias)
    return local

#loss function
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

##Function Running Part(Classification Part)
#add CNN hidden layer
n_convCNN1 = 64
n_convCNN2 = 128
n_convCNN3 = 256
output_CNN1 = add_CNN_layer(image_holder2,cNNinput,n_convCNN1,conv_size=3,stddev = 0.05,weightloss=0.0)
output_CNN2 = add_CNN_layer(output_CNN1,n_convCNN1,n_convCNN2,conv_size=3,stddev = 0.05,weightloss=0.0)
output_CNN3 = add_CNN_layer(output_CNN2,n_convCNN2,n_convCNN3,conv_size=3,stddev = 0.05,weightloss=0.0)

#add fc hidden layer
reshape = tf.reshape(output_CNN3, [batch_size, -1])
dim = reshape.get_shape()[1].value
fc1_shape = 768
fc2_shape = 192
fc3_shape = 10
output_fc1 = add_fc_layer(reshape, dim, fc1_shape,const=0.1,stddev=0.04,weightloss=0.04)
h_fc1_drop = tf.nn.dropout(output_fc1,keep_prob)
output_fc2 = add_fc_layer(h_fc1_drop, fc1_shape, fc2_shape,const=0.1,stddev=0.04,weightloss=0.04)
h_fc1_drop2 = tf.nn.dropout(output_fc2,keep_prob2)
output_fc3 = add_fc_layer(h_fc1_drop2, fc2_shape, fc3_shape,const=0.0,stddev=1 /fc2_shape ,weightloss=0.0)

#loss optimizer
loss = loss(output_fc3, label_holder2)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
top_k_op = tf.nn.in_top_k(output_fc3, tf.cast(label_holder2, tf.int64), 1)

#Initialize session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

#Training for CNN
train_step = 20000
display_step = 100
for step in range(train_step):
    start_time = time.time()
    #input train data from file
    image_batch, label_batch = sess.run([images_train, labels_train])
    #input Autoencoder dencoder layer output to CNN
    image_batch2 = sess.run(encoder_out, feed_dict={image_holder: image_batch})
    #training
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder2: image_batch2,
                                                          label_holder2: label_batch,
                                                          keep_prob:0.5, keep_prob2:0.5})
    duration = time.time() - start_time
    if step % display_step == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

#calculate precision for each class
def class_precision(pred,ground_truth, a):
    a[1][ground_truth] = a[1][ground_truth] + 1
    if pred:
        a[0][ground_truth] = a[0][ground_truth]+1
    return a
prec_matrix = np.zeros((2,10))

#Testing for CNN
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    #input test data from file
    image_batch, label_batch = sess.run([images_test, labels_test])
    #input Autoencoder encoder layer output to CNN
    image_batch2 = sess.run(encoder_out, feed_dict={image_holder: image_batch})
    predictions,pred_result = sess.run([top_k_op,output_fc3], feed_dict={image_holder2: image_batch2,
                                                                         label_holder2: label_batch,
                                                                         keep_prob:1, keep_prob2:1})
    #precision for all classes
    true_count += np.sum(predictions)
    label_one_axis = np.reshape(label_batch, batch_size)
    #precision for each class
    for i in range(batch_size):
        prec_matrix = class_precision(predictions[i],label_batch[i], prec_matrix)
    step += 1
    if step % 10 == 0:
        print(true_count)
#print precision
precision = float(true_count) / total_sample_count
print('precision @ 1 =%.3f' % precision)
print(prec_matrix)
########################################################################################################################

