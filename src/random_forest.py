import numpy as np 
import tensorflow as tf
import preprocesser
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

def run(trainset, testset,
            num_steps=500, 
            batch_sizes=1024,
            num_classes=10,
            num_features=784,
            num_trees=20,
            max_nodes=1000):
        
    #input data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    #label
    Y = tf.placeholder(tf.int32, shape=[None])

    #pass random forest params and fill the others
    fparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                            num_features=num_features,
                                            num_trees=num_trees,
                                            max_nodes=max_nodes).fill()
        
    #builf rf
    fgraph = tensor_forest.RandomForestGraphs(fparams)
    train_op = fgraph.training_graph(X, Y)
    loss_op = fgraph.training_loss(X, Y)

    #accuracy
    infer_op,_,_ = fgraph.inference_graph(X)
    correct_predic = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y,tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_predic, tf.float32))

    init = tf.group(tf.global_variables_initializer(), 
            resources.initialize_resources(resources.shared_resources()))
        
    # start session
    with tf.Session() as sess:
        sess.run(init)
        #train
        for i in range(1, num_steps + 1):
            batch_x, batch_y = trainset.next_batch(batch_sizes)
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            if i%50 == 0 or i == 1:
                acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y:batch_y})
                print("Step %i, loss: %f, acc: %f" % (i, loss, acc))
            
        #save model
        print("saving")
        saver = tf.train.Saver()
        saver.save(sess, "../model/rf/rf_model")
        print("completed")

    with tf.Session() as sess:
        saver.restore(sess, "../model/rf/rf_model")
        test_x, test_y = testset.images, testset.labels
        print("Test accuracy: ", sess.run(accuracy_op, feed_dict={X:test_x, Y:test_y}))
        
if __name__=="__main__":
    trainset, testset = preprocesser.process()
    run(trainset, testset)

