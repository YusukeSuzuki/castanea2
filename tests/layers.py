import unittest
import tensorflow as tf
import castanea2.layers as layers

class TestLayers(unittest.TestCase):
    def test_conv2d(self):
        x = tf.placeholder(tf.float32, shape=(32, 128, 128, 3))
        y = layers.conv2d(x, 3, 32)

    def test_equalized_conv2d(self):
        x = tf.placeholder(tf.float32, shape=(32, 128, 128, 3))
        y = layers.equalized_conv2d(x, 3, 32)

if __name__ == '__main__':
    unittest.main()

