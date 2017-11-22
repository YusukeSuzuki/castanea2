import tensorflow as tf


class WithNone:
    def __enter__(self): pass
    def __exit__(self,t,v,tb): pass


def device_or_none(x):
    return WithNone() if x is None else tf.device(x)


def get_conv2d_kernel_size(kernel_size):
    if type(kernel_size) in [list, tuple] and len(kernel_size) == 2:
        ks = [kernel_size[0], kernel_size[1]]
    elif type(kernel_size) == int:
        ks = [kernel_size, kernel_size]
    else:
        raise ValueError('invalid kernel_size {}.'.format(kernel_size))

    return ks
    
