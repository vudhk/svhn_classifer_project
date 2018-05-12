
from tensorflow.examples.tutorials.mnist import input_data

def process():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    return mnist.train, mnist.test

