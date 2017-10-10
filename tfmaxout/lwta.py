import numpy as np
import tensorflow as tf

"""

Maxout OP from https://arxiv.org/abs/1302.4389

Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.

Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


def lwta(inputs, num_units=None, axis=None):
    shape = inputs.get_shape().as_list()

    if num_units == None:
        num_units = inputs.get_shape().as_list()[-1]//2
        #print("NUM UNITS", num_units)


    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]

    #print "GOTCHA MY INPUTS >>>>>>>>>>>>>   ", inputs.eval()
    #print "This is shape:",shape, " And inputs shape: ", inputs.get_shape()
    print(inputs.get_shape())
    y = tf.reshape(inputs, shape)
    ind_max = tf.argmax(y, axis=-1)
    print(ind_max.get_shape().as_list())
    ind_max = tf.reshape(ind_max, [np.multiply(ind_max.get_shape().as_list()[0], ind_max.get_shape().as_list()[1])])
    MAX = tf.reduce_max(y, axis=-1)
    maxShape = MAX.get_shape().as_list()
    MAX = tf.reshape(MAX,[ np.multiply(maxShape[0], maxShape[1])])

    #print "After shape",  y.get_shape()
    # N.B. Handles 2-D case only.
    #print(ind_max.eval())
    print(y.get_shape())
    #print(inputs.eval())
    yShape = y.get_shape().as_list()
    flat_ind_max = ind_max + tf.cast(tf.range(yShape[0] * yShape[1]) * yShape[2], tf.int64)
    print("THAAA SHIT MAX SHAPE", MAX.get_shape())
    print("MY FUCKING FLAT INDEXES BEFORE", flat_ind_max.get_shape())
    flat_ind_max = tf.reshape(flat_ind_max,[flat_ind_max.get_shape().as_list()[-1], 1])
    print("MY FUCKING FLAT INDEXES AFTER", flat_ind_max.get_shape())
    print("MY FUCKING MAX INDEXES AFTER", MAX.get_shape())

    #print "I AM NOT SURE ABOUT THIS", flat_ind_max.eval()

    inShape = inputs.get_shape().as_list()
    print(inShape)
    outputs = tf.scatter_nd(flat_ind_max, MAX, [inShape[0]*inShape[1]])
    #print(outputs.eval(), outputs.get_shape())
    outputs = tf.reshape(outputs, inShape)
    return outputs


if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.Variable(np.random.uniform(size=(2, 4)))
        y = tf.square(x)
        sess.run(tf.global_variables_initializer())
        mo = lwta(x)

        print(mo.eval())

        ### THIS IS A NUMPY TEST
        """blocks = 2
        X = np.random.randn(10)
        X_C = np.array(np.reshape(X, (5, 2)))
        M = np.argmin(X_C, axis=1)
        INDICES = [np.multiply(i,2)+layer for i, layer in enumerate(M)]
        X_NEW = X_C.reshape(10)
        X_NEW[INDICES] = 0
        print "INDICES",INDICES
        print  X, " and X NEW ", X_NEW

        MASK = np.ones(shape=(5, 2))

        print "This is shit", MASK
"""
