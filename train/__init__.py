import tensorflow as tf


def average_gradients(tower_grads):
    average_grads = []

    with tf.name_scope('average_gradients'):
        for grad_and_vars in zip(*tower_grads):
            grads = []

            for g, u in grad_and_vars:
                expanded_g = tf.expand_dims(g,0)
                grads.append(expanded_g)
                tf.summary.histogram(u.name, g,
                    collections=[tf.GraphKeys.SUMMARIES, 'histogram', 'average_gradients'])

            grad = tf.concat(axis=0,values=grads)
            grad = tf.reduce_mean(grad,0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


