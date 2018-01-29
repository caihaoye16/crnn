import tensorflow as tf
import numpy as np
import math
# from net import custom_layers
from collections import namedtuple
from tensorflow.contrib import rnn
import resnet

slim = tf.contrib.slim

# =========================================================================== #
# RCNN class definition.
# =========================================================================== #
RCNNParams = namedtuple('RCNNParameters', ['ks',#kernel_size
                                         'ps',#padding_size
                                         'ss',#stride_size
                                         'nm',#In/Out size
                                         'imgH',
                                         'nc',
                                         'nclass',
                                         'nh',
                                         'n_rnn',
                                         'leakyRelu',
                                         'batch_size',
                                         'seq_length',
                                         'input_size',
                                         "reuse"
                                         ])


class CRNNNet(object):
    """

    """
    default_params = RCNNParams(
        ks=[3, 3, 3, 3, 3, 3, 2],  # kernel_size
        ps = [1, 1, 1, 1, 1, 1, 0], # padding_size
        ss = [1, 1, 1, 1, 1, 1, 1],  # stride_size
        nm = [64, 128, 256, 256, 512, 512, 512],# In/Out size
        leakyRelu = False,
        n_rnn =2,
        nh = 100,#size of the lstm hidden state
        imgH = 64,#the height / width of the input image to network
        nc = 1,
        nclass = 95-26+1,#
        batch_size= 32,
        seq_length = 26,
        input_size = 512,
        reuse = None
           )





    def __init__(self, params=None):
        """Init the net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, RCNNParams):
            self.params = params
        else:
            self.params = CRNNNet.default_params

    # ======================================================================= #
    def net(self, inputs, img_width, is_training, kp, width = None):
        """rcnn  network definition.
        """

        def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
            shape = input_.get_shape().as_list()

            with tf.variable_scope(scope or "Linear"):
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.truncated_normal_initializer(stddev=stddev))
                bias = tf.get_variable("bias", [output_size],
                    initializer=tf.constant_initializer(bias_start))
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias

        def BLSTM(inputs, n_hidden, num_layers, num_classes, scope='blstm'):
            # # Defining the cell
            # # Can be:
            # #   tf.nn.rnn_cell.RNNCell
            # #   tf.nn.rnn_cell.GRUCell
            # cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

            # # Stacking rnn cells
            # stack = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers,
            #                                                   state_is_tuple=True)

            # # The second output is the last state and we will no use that
            # outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_len, dtype=tf.float32)#seq_len

            # shape = tf.shape(inputs)
            # batch_s, max_timesteps = shape[0], shape[1]

            # # Reshaping to apply the same weights over the timesteps
            # outputs = tf.reshape(outputs, [-1, num_hidden])

            # # Truncated normal with mean 0 and stdev=0.1
            # # Tip: Try another initialization
            # # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            # W = tf.Variable(tf.truncated_normal([num_hidden,
            #                                      num_classes],
            #                                     stddev=0.1), name="W")
            # # Zero initialization
            # # Tip: Is tf.zeros_initializer the same?
            # b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

            # # Doing the affine projection
            # logits = tf.matmul(outputs, W) + b

            # # Reshaping back to the original shape
            # logits = tf.reshape(logits, [batch_s, -1, num_classes])

            # # Time major
            # logits = tf.transpose(logits, (1, 0, 2))
            # return logits, inputs, seq_len, W, b


            with tf.variable_scope(scope):
                outputs = inputs
                for i in range(num_layers):
                    with tf.variable_scope('layer_%d' %i):
                        # Define lstm cells with tensorflow
                        # Forward direction cell
                        lstm_fw_cell = rnn.LSTMBlockCell(n_hidden)
                        # Backward direction cell
                        lstm_bw_cell = rnn.LSTMBlockCell(n_hidden)

                        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs, dtype=tf.float32)
                        outputs = tf.concat(outputs, 2)

                output_list = tf.unstack(outputs, axis=1)
                final_output = []
                for output in output_list:

                    final_output.append(linear(output, num_classes, 'output_layer'))

                    tf.get_variable_scope().reuse_variables()
            
                final_output = tf.stack(final_output, 1)

            return final_output


        def conv2d(inputs, nOut, kernel_size, stride=1, padding="SAME", batchNormalization=False, is_training=None, scope='conv2d'):
            # nOut = self.params.nm[i]
            # kernel_size = self.params.ks[i]
            # stride = self.params.ss[i]
            
            net = slim.conv2d(inputs, nOut, kernel_size, stride=stride, padding=padding, scope=scope, activation_fn=None)

            if batchNormalization:
                net = slim.batch_norm(net, is_training=is_training, scope=scope+'_bn')

            net = tf.nn.relu(net)
            return net

        def feature_extractor(inputs):
            net = conv2d(inputs, 64, 3, scope='conv1')#input batch_size*32*100*3 #net batch_size *32*100*64
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')#net batch_size *16*50*64
            print("pool_0 ", net.shape)
            #batch_size*16*50*128
            net = conv2d(net, 128, 3, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')#batch_size *8*25*128
            print("pool_1 ", net.shape)
            net = conv2d(net, 256, 3, batchNormalization=True, is_training=is_training, scope='conv3')#b *8*25*256
            net = conv2d(net, 256, 3, scope='conv4') #b*8*25*256
            # net = custom_layers.pad2d(net, pad=(0, 1))#b*8*27*256
            net = slim.max_pool2d(net, [2, 2], stride =[2,1], padding='SAME', scope='pool3')#b*4*26*256
            print('pool_3', net.shape)
            #net = slim.max_pool2d(net,[2,2],stride =[2,1],padding ="SAME")
            net = conv2d(net, 512, 3, batchNormalization=True, is_training=is_training, scope='conv5') #b*4*26*512

            net = tf.nn.dropout(net, kp)

            net = conv2d(net, 512, 3, scope='conv6') #b*4*26*512

            net = tf.nn.dropout(net, kp)

            # net = custom_layers.pad2d(net, pad=(0, 1))#b*4*28*512
            net = slim.max_pool2d(net, [2, 2], stride =[2,1], padding='SAME', scope='pool4') #b*2*27*512
            print("pool_4", net.shape)
            net = conv2d(net, 512, 2, batchNormalization=True, is_training=is_training, padding='VALID', scope='conv7') #b*1*26*512
            print("conv7",net.shape)

            net = tf.nn.dropout(net, kp)

            return net

        with tf.variable_scope('ResNet', reuse=None):
            model = resnet.imagenet_resnet_v2(50, None, 'channels_last')
            net = model(inputs=inputs, is_training=is_training)
            print("net shape: ", net.shape)

        with tf.variable_scope("CRNN_net",reuse=None):

            # net = feature_extractor(inputs)

            net = tf.squeeze(net,[1])#B*26*512
            print("squeeze: ", net.shape)

            batch_size, length, _ = net.shape.as_list()
            seq_len = np.full(batch_size, length)

            # # get sequence lengths
            # # size after first pooling
            # # ceil(float(in_height) / float(strides[1]))
            # img_width = tf.ceil(tf.cast(img_width, tf.float32) / 2.)
            # # size after second pooling
            # img_width = tf.ceil(tf.cast(img_width, tf.float32) / 2.)
            # seq_len = tf.cast(img_width - 1, tf.int32) # seq_len can be used to mask out the wasted length (meanwhile returned for ctc_loss calculation)

            logits = BLSTM(net, 256, 2, self.params.nclass)

            return logits, seq_len


            # # print("transpose: ", net.shape)
            # if width is None:
            #     seq_length = self.params.seq_length
            # else:
            #     seq_length = tf.cast(width/4+1,tf.int32)
            # seq_len = np.ones(self.params.batch_size) * seq_length

            # return BLSTM(net,self.params.nh,2,seq_len,self.params.nclass)



    def losses(self, targets, logits, seq_len,
               scope='ctc_losses'):
        """Define the network losses.
        """
        with tf.control_dependencies([tf.less_equal(targets.dense_shape[1], tf.reduce_max(tf.cast(seq_len, tf.int64)))]):
            with tf.name_scope(scope):
                loss = tf.nn.ctc_loss(targets, logits, seq_len, time_major=False, ignore_longer_outputs_than_inputs=True)
                cost = tf.reduce_mean(loss)

        return cost
