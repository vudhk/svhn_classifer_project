
# coding: utf-8

# In[1]:


import scipy.io as io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import traceback
from tensorflow.python.ops import nn


# In[2]:


# dinh nghia cac tham so mac dinh
VALIDATION_SIZE = 14652
TRAINING_INTERS = 30
BATCH_SIZE = 200
N_INPUT = 32
N_CLASSES = 10
N_CHANNEL = 3


# In[3]:


def prepare_data():
    train_data = io.loadmat('../svhn_data/train_32x32.mat', variable_names='X').get('X').transpose([3,0,1,2])
    train_label = io.loadmat('../svhn_data/train_32x32.mat', variable_names='y').get('y')
    test_data = io.loadmat('../svhn_data/test_32x32.mat', variable_names='X').get('X').transpose([3,0,1,2])
    test_label = tf.one_hot(np.squeeze(io.loadmat('../svhn_data/test_32x32.mat', variable_names='y').get('y')), N_CLASSES)
    validation_data = train_data[:VALIDATION_SIZE]
    validation_label = tf.one_hot(np.squeeze(train_label[:VALIDATION_SIZE]), N_CLASSES)
    train_data = train_data[VALIDATION_SIZE:]
    train_label = tf.one_hot(np.squeeze(train_label[VALIDATION_SIZE:]), N_CLASSES)
    
    with tf.Session() as sess:
        train_label, validation_label, test_label = sess.run([train_label, validation_label, test_label])

    
    return {'data': train_data, 'label': train_label},{'data': validation_data, 'label': validation_label},{'data': test_data, 'label': test_label}


# In[4]:


# Xay dung mo hinh cnn su dung slim
def cnn_1(input, is_training=True):
    # su dung arg_scope de chac chan rang moi layer chi su dung cung mot gia tri cua cac parameter
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                            normalizer_fn=slim.batch_norm):
    
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            
            tf.summary.histogram('input', input)
            tf.summary.image('input', input[:,:,:,:3])
            
            # Convolution layer 1
            net = slim.conv2d(input, 32, 5, scope='conv1')
            tf.summary.histogram('conv1', net)
            
            net = slim.max_pool2d(net, 2, scope='pool1')
            tf.summary.image('pool1', net[:,:,:,:3])
            
            # Convolution layer 2
            net = slim.conv2d(net, 64, 5, scope='conv2')
            tf.summary.histogram('conv2', net)
            
            net = slim.max_pool2d(net, 2, scope='pool2')
            tf.summary.image('pool2', net[:,:,:,:3])
            
            # Convolution layer 3
            net = slim.conv2d(net, 128, 5, scope='conv3')
            tf.summary.histogram('conv3', net)
            
            net = slim.max_pool2d(net, 2, scope='pool3')
            tf.summary.image('pool3', net[:,:,:,:3])
            
            net = slim.flatten(net, scope='flatten1')
            
        # Fully connected layer 1
        net = slim.fully_connected(net, 1024, scope='fc1')
        tf.summary.histogram('fc1', net)
        net = slim.dropout(net, is_training=is_training, scope='dropout1')  # 0.5 by default
        
        # Fully connected layer 1
        outputs = slim.fully_connected(net, N_CLASSES, activation_fn=None, scope='fco')
        tf.summary.histogram('fco', net)
        
    return outputs


# In[5]:


# Building Re-shaped AlexNet
def cnn_2(input, is_training=True):
    # su dung arg_scope de chac chan rang moi layer chi su dung cung mot gia tri cua cac parameter
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                            normalizer_fn=slim.batch_norm):
    
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            
            tf.summary.histogram('input', input)
            tf.summary.image('input', input[:,:,:,:3])
            
            # Convolution layer 1
            net = slim.conv2d(input, 48, 8, stride=3, scope='conv1')
            tf.summary.histogram('conv1', net)
            
            net = slim.max_pool2d(net, 2, stride=1, scope='pool1')
            tf.summary.image('pool1', net[:,:,:,:3])
            
            net = tf.nn.local_response_normalization(net)
            
            # Convolution layer 2
            net = slim.conv2d(net, 126, 2, stride=1, scope='conv2')
            tf.summary.histogram('conv2', net)
            
            net = slim.max_pool2d(net, 2, stride=1, scope='pool2')
            tf.summary.image('pool2', net[:,:,:,:3])
            
            net = tf.nn.local_response_normalization(net)
            
            # Convolution layer 3
            net = slim.conv2d(net, 192, 3, stride=1, scope='conv3')
            tf.summary.histogram('conv3', net)
            
            # Convolution layer 4
            net = slim.conv2d(net, 192, 2, stride=1, scope='conv4')
            tf.summary.histogram('conv4', net)
            
            # Convolution layer 5
            net = slim.conv2d(net, 128, 2, stride=1, scope='conv5')
            tf.summary.histogram('conv5', net)
            
            net = slim.max_pool2d(net, 2, stride=1, scope='pool3')
            tf.summary.image('pool3', net[:,:,:,:3])
            
            net = tf.nn.local_response_normalization(net)
            
            net = slim.flatten(net, scope='flatten1')
            
        # Fully connected layer 1
        net = slim.fully_connected(net, 1024, scope='fc1')
        tf.summary.histogram('fc1', net)
        net = slim.dropout(net, is_training=is_training, scope='dropout1')  # 0.5 by default
        
         # Fully connected layer 2
        net = slim.fully_connected(net, 1024,scope='fc2')
        tf.summary.histogram('fc2', net)
        net = slim.dropout(net, is_training=is_training, scope='dropout2')
        
         # Fully connected layer 3
        outputs = slim.fully_connected(net, N_CLASSES, activation_fn=nn.softmax, scope='fco')
        tf.summary.histogram('fco', net)
        
    return outputs


# In[6]:


def train():
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init, feed_dict={is_training:True})
        # Training cycle
        for epoch in range(TRAINING_INTERS):
            total_batch =  train_len // BATCH_SIZE
            for batch in range(total_batch):
                # lay batch tiep theo
                batch_input = train_set['data'][batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,train_len)]
                batch_label = train_set['label'][batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,train_len)]
                # chay train_op, loss_op, accuracy
                _, cost, acc, summary = sess.run([train_op, loss_op, accuracy, merged_summary_op], feed_dict={input:batch_input, label:batch_label, is_training:True})
                # Write logs at every iteration
                writer.add_summary(summary, epoch * total_batch + batch)

            # hien thi ket qua sau moi epoch
            file_writer.write("Epoch:" + ('%04d,' % (epoch + 1)) + ("cost={%.9f}, training accuracy %.5f" % (cost, acc)) + "\n")
            file_writer.flush()

        file_writer.write('Optimization completed!!!\n')

        # Luu tru variables vao disk.
        save_path = saver.save(sess, model_path)
        file_writer.write("Model saved in path: %s \n" % save_path)
        file_writer.flush()
        


# In[7]:


def evalute(dataset, model_path):
    dataset_len = dataset['data'].shape[0]
    # Su dung model da duoc luu de tien doan
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        avg_acc = 0.
        total_batch =  dataset_len // BATCH_SIZE
        for batch in range(total_batch):
            # lay batch tiep theo
            batch_input = dataset['data'][batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,dataset_len)]
            batch_label = dataset['label'][batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,dataset_len)]    
            acc = sess.run(accuracy, feed_dict={input:batch_input, label:batch_label, is_training:False})
            avg_acc += acc / total_batch
            
        file_writer.write("Accuracy on test set: %.5f \n" % (avg_acc))
        file_writer.flush()
        
    return avg_acc


# In[8]:


if __name__ == '__main__':

    file_writer = open('logfile.log', 'w')

    # Lay va phan chia data tu file
    train_set, validation_set, test_set = prepare_data()
    train_len = train_set['data'].shape[0]
    test_len = test_set['data'].shape[0]
    validation_len = validation_set['data'].shape[0]

    accs = []
    try:
        for i in range(1,3):
            tf.reset_default_graph()
            is_training = tf.placeholder(tf.bool, name='is_training')
            # graph input
            input = tf.placeholder(tf.float32, [None, N_INPUT,N_INPUT,N_CHANNEL], name='train_input')
            # graph label
            label = tf.placeholder(tf.float32, [None, N_CLASSES], name='train_label')

            # predict
            if i == 1:
                file_writer.write('Test model of cnn_1 net....\n')
                file_writer.flush()
                pred = cnn_1(input, is_training)
            elif i == 2:
                file_writer.write('Test model of cnn_2 net....\n')
                file_writer.flush()
                pred = cnn_2(input, is_training)
            else:
                raise ValueError('Out of model')

            # tu dong tim kiem learning_rate phu hop
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                        1e-4,  # Base learning rate.
                        global_step * BATCH_SIZE,  # Current index into the dataset.
                        train_len,  # Decay step.
                        0.95,  # Decay rate.
                        staircase=True)

            # dinh nghia cac ham mat mat
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label))
            optimal = tf.train.AdamOptimizer(learning_rate)
            train_op = optimal.minimize(loss_op, global_step=global_step)

            # danh gia model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # tao summary cua cac monitor de quan sat cac bien
            tf.summary.scalar('loss_op', loss_op)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('accuracy', accuracy)

            # gop cac summaries vao mot operation
            merged_summary_op = tf.summary.merge_all()

            # tao doi tuong log writer va ghi vao Tensorboard
            writer = tf.summary.FileWriter('./checkpoint', graph=tf.get_default_graph())

            # khoi tao cac variables
            init = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            model_path = "model_" + str(i) + "/cnn_model.ckpt"

            # training
            start = time.time()
            train()
            elapsed = time.time() - start
            file_writer.write('--- Training process is completed in %.3f (s) \n' % (elapsed))
            file_writer.flush()
            
            # test on validation set
            start = time.time()
            acc = evalute(validation_set, model_path)
            elapsed = time.time() - start
            file_writer.write('--- Evalution process on validate set is completed in %.3f (s) \n' % (elapsed))
            file_writer.write('-----------------------------------------------------------------')
            file_writer.flush()
            
            accs.append(acc)


        # test the best model on test set
        best_model = accs.index(max(accs)) + 1
        start = time.time()
        model_path = "model_" + str(best_model) + "/cnn_model.ckpt"
        acc = evalute(test_set, model_path)
        elapsed = time.time() - start
        file_writer.write('--- Evalution process on test set is completed in %.3f (s) \n' % (elapsed))
        file_writer.flush()
        
    except:
        file_writer.write(traceback.format_exc())
        file_writer.flush()
        
    file_writer.close()
    print('Done')

