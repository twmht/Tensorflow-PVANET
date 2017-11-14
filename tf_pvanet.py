import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MOMENTUM = 0.9
RESNET_VARIABLES = 'pvanet_variables'
UPDATE_OPS_COLLECTION = 'pvanet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

logs_path = '/tmp/pvanet_train'


class PVANET():
    def __init__(self, is_training=True):
        if is_training:
            use_global_stats = False
            batch_size = 128
        else:
            use_global_stats = True
            batch_size = 1

        self.is_training = tf.convert_to_tensor(is_training,
                                           dtype='bool',
                                           name='is_training')


        x = tf.placeholder(tf.float32, [1, 320, 320, 3], name = 'input')
        with tf.variable_scope('conv_1_1'):
            # 7*7 C.ReLU
            x = self.build_c_relu_block(x, kernel=7, stride=2, filters_out=16, to_seperable=False)
        # max pool 3_3
        with tf.variable_scope('pool1_1'):
            x =  tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

        with tf.variable_scope('conv_2_1'):
            #  # 7*7 C.ReLU
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[24, 24, 64], to_seperable=False, residual=dict(kernel=1, stride=1, filters_out=64))


        with tf.variable_scope('conv_2_2'):
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[24, 24, 64], to_seperable=False, residual=dict(kernel=1, stride=1, filters_out=64), bn_input=True)

        with tf.variable_scope('conv_2_3'):
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[24, 24, 64], to_seperable=False, residual=dict(kernel=1, stride=1, filters_out=64), bn_input=True)


        with tf.variable_scope('conv_3_1'):
            x = self._bn(x)
            x = self._relu(x)
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[2, 1, 1], filters_out=[48, 48, 128], to_seperable=False, residual=dict(kernel=1, stride=2, filters_out=128))



        with tf.variable_scope('conv_3_2'):
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[48, 48, 128], to_seperable=False, bn_input=True)

        with tf.variable_scope('conv_3_3'):
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[48, 48, 128], to_seperable=False, bn_input=True)


        with tf.variable_scope('conv_3_4'):
            x = self.build_residual_c_rulu_block(x, kernel=[1, 3, 1], stride=[1, 1, 1], filters_out=[48, 48, 128], to_seperable=False, bn_input=True)
            conv_3_4 = x

        with tf.variable_scope('conv_4_1'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=64)


            #  [48, 128]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=48)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=128, pad='SAME')


            # [24, 48, 48]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=24)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            # pool
            with tf.variable_scope('incep_pool_0'):
                incep_pool =  tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')

            with tf.variable_scope('incep_pool_1'):
                incep_pool = self.build_conv_bn(incep_pool, kernel=1, stride=1, filters_out=128 )

            concat = tf.concat([incep_0, incep_1, incep_2, incep_pool], axis=3)
            x = self._conv(concat, kernel=1, stride=1, filters_out=256)

            with tf.variable_scope('proj'):
                proj = self._conv(conv_3_4, kernel=1, stride=2, filters_out=256)
            conv_4_1 = x + proj
            x = conv_4_1

        with tf.variable_scope('conv_4_2'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [64, 128]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=128, pad='SAME')


            # [24, 48, 48]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=24)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)
            x = self._conv(concat, kernel=1, stride=1, filters_out=256)

            conv_4_2 = x + conv_4_1
            x = conv_4_2
            assert(conv_4_2.get_shape().as_list() == [1, 20, 20, 256])

        with tf.variable_scope('conv_4_3'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [64, 128]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=128, pad='SAME')


            # [24, 48, 48]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=24)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)
            x = self._conv(concat, kernel=1, stride=1, filters_out=256)

            conv_4_3 = x + conv_4_2
            x = conv_4_3
            assert(conv_4_3.get_shape().as_list() == [1, 20, 20, 256])

        with tf.variable_scope('conv_4_4'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [64, 128]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=128, pad='SAME')


            # [24, 48, 48]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=24)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=48, pad='SAME')

            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)
            x = self._conv(concat, kernel=1, stride=1, filters_out=256)

            conv_4_4 = x + conv_4_3
            x = conv_4_4
            assert(conv_4_3.get_shape().as_list() == [1, 20, 20, 256])

        with tf.variable_scope('conv_5_1'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=64)


            #  [96, 192]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=96)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=192, pad='SAME')


            # [32, 64, 64]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=2, filters_out=32)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')

            # pool
            with tf.variable_scope('incep_pool_0'):
                incep_pool =  tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('incep_pool_1'):
                incep_pool = self.build_conv_bn(incep_pool, kernel=1, stride=1, filters_out=128 )

            concat = tf.concat([incep_0, incep_1, incep_2, incep_pool], axis=3)

            assert (concat.get_shape().as_list() == [1, 10, 10, 448])

            x = self._conv(concat, kernel=1, stride=1, filters_out=384)

            with tf.variable_scope('proj'):
                proj = self._conv(conv_4_4, kernel=1, stride=2, filters_out=384)
            conv_5_1 = x + proj
            x = conv_5_1
            assert (conv_5_1.get_shape().as_list() == [1, 10, 10, 384])

        with tf.variable_scope('conv_5_2'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [96, 192]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=96)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=192, pad='SAME')


            # [32, 64, 64]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=32)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')


            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)

            assert (concat.get_shape().as_list() == [1, 10, 10, 320])

            x = self._conv(concat, kernel=1, stride=1, filters_out=384)

            conv_5_2 = x + conv_5_1
            x = conv_5_2
            assert (conv_5_1.get_shape().as_list() == [1, 10, 10, 384])

        with tf.variable_scope('conv_5_3'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [96, 192]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=96)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=192, pad='SAME')


            # [32, 64, 64]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=32)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')


            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)

            assert (concat.get_shape().as_list() == [1, 10, 10, 320])

            x = self._conv(concat, kernel=1, stride=1, filters_out=384)

            conv_5_3 = x + conv_5_2
            x = conv_5_3
            assert (conv_5_1.get_shape().as_list() == [1, 10, 10, 384])

        with tf.variable_scope('conv_5_4'):
            x = self._bn(x)
            x = self._relu(x)
            #  [64]
            with tf.variable_scope('incep_0'):
                incep_0 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=64)


            #  [96, 192]
            with tf.variable_scope('incep_1_0'):
                incep_1 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=96)

            with tf.variable_scope('incep_1_1'):
                incep_1 = self.build_conv_bn(incep_1, kernel=3, stride=1, filters_out=192, pad='SAME')


            # [32, 64, 64]
            with tf.variable_scope('incep_2_0'):
                incep_2 = self.build_conv_bn(x, kernel=1, stride=1, filters_out=32)

            with tf.variable_scope('incep_2_1'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')

            with tf.variable_scope('incep_2_2'):
                incep_2 = self.build_conv_bn(incep_2, kernel=3, stride=1, filters_out=64, pad='SAME')


            concat = tf.concat([incep_0, incep_1, incep_2], axis=3)

            assert (concat.get_shape().as_list() == [1, 10, 10, 320])

            with tf.variable_scope('out'):
                x = self.build_conv_bn(concat, kernel=1, stride=1, filters_out=384)

            conv_5_4 = x + conv_5_3
            assert (conv_5_4.get_shape().as_list() == [1, 10, 10, 384])

            with tf.variable_scope('last_bn'):
                x = self._bn(conv_5_4)
                x = self._relu(x)


        with tf.variable_scope('pool5'):
            x =  tf.nn.max_pool(x, ksize=[1, 1, 1, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')


        with tf.variable_scope('fc6'):
            x = tf.contrib.layers.fully_connected(x, 4096)
            x = self._bn(x)
            x = tf.contrib.layers.dropout(x)
            x = self._relu(x)

        with tf.variable_scope('fc7'):
            x = tf.contrib.layers.fully_connected(x, 4096)
            x = self._bn(x)
            x = tf.contrib.layers.dropout(x)
            x = self._relu(x)

        with tf.variable_scope('fc8'):
            x = tf.contrib.layers.fully_connected(x, 1000)

    # each crelu has 1 X 1 convolution
    def build_conv_bn(self, x, kernel, stride, filters_out, pad='SAME'):
        x = self._conv(x, kernel=kernel, stride=stride, filters_out=filters_out, padding=pad)
        x = self._bn(x)
        x = self._relu(x)
        return x

    def build_c_relu_block(self, x, kernel, stride, filters_out, to_seperable=False):
        # convolution
        in_channel = x.shape[-1]
        if to_seperable:
            conv = get_seperable(x, kernel, stride, filters_out)
        else:
            x = self._conv(x, kernel=kernel, stride=stride, filters_out=filters_out)
            x = self._bn(x)
        #  batch normalization
        with tf.variable_scope('CRELU'):
            x = tf.concat([x, -x], -1)
            # FIXME: scale and shift
            x_shape = x.get_shape()
            params_shape = x_shape[-1:]
            scale = self._get_variable('scale',
                                 params_shape,
                                 initializer=tf.zeros_initializer())
            shift = self._get_variable('shift',
                                  params_shape,
                                  initializer=tf.ones_initializer())
            x = tf.add(tf.multiply(x, scale), shift)
            y = self._relu(x)

        return y

    def get_name_scope(self):
        return tf.contrib.framework.get_name_scope();

    def build_residual_c_rulu_block(self, x, kernel, stride, filters_out, to_seperable=False, residual=dict(), bn_input=False):
        layer_in = x

        if bn_input:
            x = self._bn(x)
            x = self._relu(x)

        with tf.variable_scope('block_1'):
            x = self.build_conv_bn(x, kernel=kernel[0], stride=stride[0], filters_out=filters_out[0])
        # c.relu block
        with tf.variable_scope('block_2'):
            x = self.build_c_relu_block(x, kernel=kernel[1], stride=stride[1], filters_out=filters_out[1])
        # 1 X 1 convoltion
        with tf.variable_scope('block_3'):
            x = self._conv(x, kernel[2], stride[2], filters_out[2])

        if residual:
            layer_in = self._conv(layer_in, kernel=residual['kernel'], stride=residual['stride'], filters_out=residual['filters_out'])

        return x + layer_in

    def get_seperable(x, kernel, stride, filters_out):
        filters_in = x.get_shape()[-1]
        assert(kernel > 0 and kernel > 0)
        # FIXME: pad for tensorflow?
        # depthwise
        x = tf.nn.depthwise_conv2d(x, [kernel, kernel, in_channel, 1], [1, stride, stride, 1], padding='VALID')
        x = self._bn(x)
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        beta = self._get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer())
        gamma = self._get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer())
        x = beta * x + gamma
        x = self._relu(x)
        # pointwise
        y = self._conv(x, filters_out, 1, 1)
        return y

    def _relu(self, x):
        return tf.nn.relu(x)

    def _bn(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer())
        gamma = self._get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer())

        # FIXME: trainable?
        moving_mean = self._get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer(),
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer(),
                                        trainable=False)

        # these ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            self.is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        #x.set_shape(inputs.get_shape()) ??

        return x

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)


    def _conv(self, x, kernel, stride, filters_out, padding='SAME'):
        filters_in = x.get_shape()[-1]
        shape = [kernel, kernel, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=shape,
                                dtype='float',
                                initializer=initializer,
                                weight_decay=CONV_WEIGHT_DECAY)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)


if __name__ == '__main__':
    #  tf.reset_default_graph()
    PVANET()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
