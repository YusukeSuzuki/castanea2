import unittest
import tensorflow as tf
from layers.conv2d import conv2d

class TestLayers(unittest.TestCase):
    def test_conv2d(self):
        x = tf.placeholder(tf.float32, shape=(32, 128, 128, 3))
        y = conv2d(x, 3, 32)

if __name__ == '__main__':
    unittest.main()

